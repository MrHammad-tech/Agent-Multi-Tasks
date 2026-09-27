import os
import re
import json
import requests
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from openai import OpenAI

DEFAULT_GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)

DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("AGENT_OUTPUT_DIR", "./agent_output")
)


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, default)
    return value.strip() if isinstance(value, str) else default


AI_PROVIDER = _env("AI_PROVIDER", "gemini").lower()
AI_API_KEY = _env("AI_API_KEY")
AI_MODEL = _env("AI_MODEL")
AI_BASE_URL = _env("AI_BASE_URL", DEFAULT_GEMINI_BASE_URL)

AI_FALLBACK_PROVIDER = _env("AI_FALLBACK_PROVIDER").lower()
AI_FALLBACK_API_KEY = _env("AI_FALLBACK_API_KEY")
AI_FALLBACK_MODEL = _env("AI_FALLBACK_MODEL")
AI_FALLBACK_BASE_URL = _env("AI_FALLBACK_BASE_URL")

def _json(data: Any) -> str:
    """Return safe JSON for tool results."""
    try:
        return json.dumps(data, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps({"error": str(data)}, ensure_ascii=False)


def _error(message: str) -> str:
    return _json({"status": "error", "error": message})


def _ok(**data: Any) -> str:
    return _json({"status": "success", **data})


def _require_package(package: Any, name: str) -> Optional[str]:
    if package is None:
        return f"Required package '{name}' is not installed."
    return None


def save_training_data_atomic(
    history: List[Dict[str, Any]],
    filename: str = "agent_training_data.jsonl",
) -> bool:
    """Append one complete conversation record safely."""
    try:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        record = {"messages": history}
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            try:
                os.fsync(file.fileno())
            except OSError:
                pass
        return True
    except Exception as exc:
        print(f"[Agent] Training-data persistence failed: {exc}")
        return False


def save_training_data(
    history: List[Dict[str, Any]],
    filename: str = "agent_training_data.jsonl",
) -> str:
    if save_training_data_atomic(history, filename):
        return f"Training data saved to '{filename}'."
    return f"Could not save training data to '{filename}'."

OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://localhost:11434/v1"
)
OLLAMA_MODEL = "qwen2.5:3b"

def read_file(
    file_path: str,
    max_lines: Optional[int] = None,
    **_: Any,
) -> str:
    try:
        path = Path(file_path).expanduser()

        if not path.exists():
            return _error(f"File '{file_path}' does not exist.")
        if not path.is_file():
            return _error(f"'{file_path}' is not a file.")

        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:
            if max_lines is None:
                content = file.read()
            else:
                if max_lines < 1:
                    return _error("max_lines must be greater than zero.")
                content = "".join(
                    line for _, line in zip(range(max_lines), file)
                )

        return _ok(
            file_path=str(path.resolve()),
            content=content,
        )
    except Exception as exc:
        return _error(f"Could not read file: {exc}")

def save_file(
    file_path: str,
    content: str,
    base_dir: str = "./agent_output",
    **_: Any,
) -> str:
    try:
        base = Path(base_dir).expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)

        target = (base / file_path).resolve()

        try:
            target.relative_to(base)
        except ValueError:
            return _error("Path traversal detected. Access denied.")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")

        return _ok(
            message="File successfully saved.",
            file_path=str(target),
            bytes_written=len(str(content).encode("utf-8")),
        )
    except Exception as exc:
        return _error(f"Could not save file: {exc}")

        
def list_directory(path: str = ".", directory_path: str = None, **kwargs):
  target_dir = directory_path or path
  if not os.path.exists(target_dir):
    return json.dumps({"error": f"Directory '{target_dir}' does not exist."})

  try:
    items = os.listdir(target_dir)
    return json.dumps({"status": "success", "path": target_dir, "items": items})
  except Exception as e:
    return json.dumps({"error": str(e)})

def inspect_dataset(file_path: str):
    if not os.path.exists(file_path):
        return f"Error: {file_path} does not exists"
 
    try:
        df = pd.read_csv(file_path)
        preview = {
            'Shape': df.shape,
            'Columns': list(df.columns),
            'Head': df.head(3).to_dict()
        }
        return str(preview)
    except Exception as e:
        return f"Error: Inspecting Dataset {str(e)}"
    
def compute_statistics(file_path:str, column_name:str):
    if not os.path.exists(file_path):
        return f"Error: {file_path} does not exists"
    
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            return f"Error: {column_name} does not exists"
        col_data = df[column_name].dropna()
        stats_summary = {
            "mean": float(col_data.mean()),
            "median": float(col_data.median()),
            "variance": float(col_data.var()),
            "skewness": float(stats.skew(col_data))
        }
        return str(stats_summary)
    except Exception as e:
        return f"Error: Computing stats {str(e)}"
    
def train_model(
    file_path: str,
    target_column: str,
    task_type: str = "classification",
) -> str:
    missing = _require_package(pd, "pandas")
    if missing:
        return _error(missing)

    missing = _require_package(xgb, "xgboost")
    if missing:
        return _error(missing)

    if train_test_split is None:
        return _error("scikit-learn is not installed.")

    task_type = str(task_type).lower().strip()

    if task_type not in {"classification", "regression"}:
        return _error(
            "task_type must be 'classification' or 'regression'."
        )

    path = Path(file_path).expanduser()

    if not path.exists():
        return _error(f"Dataset '{file_path}' does not exist.")

    try:
        df = pd.read_csv(path)

        if target_column not in df.columns:
            return _error(
                f"Target column '{target_column}' is missing."
            )

        feature_frame = df.drop(columns=[target_column]).select_dtypes(
            include=["number"]
        )

        if feature_frame.empty:
            return _error(
                "No numeric feature columns are available. "
                "This version intentionally does not guess how to encode "
                "categorical features."
            )

        # Remove rows with missing numeric features/target.
        clean = pd.concat(
            [feature_frame, df[target_column]],
            axis=1,
        ).dropna()

        if len(clean) < 5:
            return _error(
                "The dataset needs at least 5 complete rows for training."
            )

        X = clean[feature_frame.columns]
        y = clean[target_column]

        stratify = None
        if task_type == "classification":
            counts = y.value_counts()
            if len(counts) < 2:
                return _error(
                    "Classification requires at least two target classes."
                )
            if counts.min() >= 2:
                stratify = y

        test_size = 0.2
        if len(clean) < 10:
            test_size = 0.4

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )

        if task_type == "regression":
            if mean_squared_error is None:
                return _error("scikit-learn metrics are unavailable.")

            model = xgb.XGBRegressor(
                n_estimators=50,
                random_state=42,
                objective="reg:squarederror",
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)

            return _ok(
                task="regression",
                rows=int(len(clean)),
                features=[str(c) for c in X.columns],
                test_mse=float(mse),
            )

        if accuracy_score is None:
            return _error("scikit-learn metrics are unavailable.")

        model = xgb.XGBClassifier(
            n_estimators=50,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        return _ok(
            task="classification",
            rows=int(len(clean)),
            features=[str(c) for c in X.columns],
            classes=[str(c) for c in model.classes_],
            test_accuracy=float(accuracy),
            test_accuracy_percent=round(float(accuracy) * 100, 2),
        )

    except Exception as exc:
        return _error(f"Model training failed: {exc}")
    
def web_scrape(url: str, max_chars: int = 10000) -> str:
    """
    Fetch a webpage and return cleaned text.

    Important: this action does NOT call an AI provider or Ollama.
    The Agent can summarize the returned text afterwards.
    """
    if requests is None:
        return _error("requests is not installed.")

    if not isinstance(url, str) or not url.strip():
        return _error("A URL is required.")

    url = url.strip()

    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        url = "https://" + url

    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120 Safari/537.36"
                )
            },
            timeout=20,
            allow_redirects=True,
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if "text" not in content_type and "html" not in content_type:
            return _error(
                f"Unsupported response type: {content_type or 'unknown'}"
            )

        html = response.text

        html = re.sub(
            r"<(script|style|noscript).*?>.*?</\1>",
            " ",
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )

        html = re.sub(r"<!--.*?-->", " ", html, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()

        if max_chars < 1:
            max_chars = 10000

        return _ok(
            url=response.url,
            status_code=response.status_code,
            text=text[:max_chars],
            truncated=len(text) > max_chars,
        )

    except requests.RequestException as exc:
        return _error(f"Web request failed: {exc}")
    except Exception as exc:
        return _error(f"Web scraping failed: {exc}")
    
dispatcher_registry: Dict[str, Callable[..., str]] = {
    "read_file": read_file,
    "save_file": save_file,
    "list_directory": list_directory,
    "inspect_dataset": inspect_dataset,
    "compute_statistics": compute_statistics,
    "train_xgboost_model": train_model,
    "web_scrape": web_scrape,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read and return the full text content of a local file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to read."}
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_file",
            "description": "Save text content to a file inside the agent's sandboxed output directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Relative path/filename to save to."},
                    "content": {"type": "string", "description": "The text content to write."},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List the files and folders inside a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory_path": {"type": "string", "description": "Path to the directory."}
                },
                "required": ["directory_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_dataset",
            "description": "Preview a CSV file's shape, columns, and first few rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the CSV file."}
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_statistics",
            "description": "Compute mean, median, variance, and skewness for one numeric column of a CSV file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the CSV file."},
                    "column_name": {"type": "string", "description": "Name of the numeric column."},
                },
                "required": ["file_path", "column_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "train_xgboost_model",
            "description": "Train an XGBoost classifier or regressor on a CSV file and report test performance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the CSV file."},
                    "target_column": {"type": "string", "description": "Column to predict."},
                    "task_type": {
                        "type": "string",
                        "enum": ["classification", "regression"],
                        "description": "Whether to train a classifier or a regressor.",
                    },
                },
                "required": ["file_path", "target_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_scrape",
            "description": "Fetch a web page and return its cleaned, tag-stripped text (first 3000 chars).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to scrape."}
                },
                "required": ["url"],
            },
        },
    },
]

def _provider_settings(
    provider: str,
    api_key: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
) -> Tuple[str, str, str]:
    provider = (provider or "gemini").lower().strip()

    if provider == "gemini":
        key = api_key or AI_API_KEY
        selected_model = model or AI_MODEL
        selected_base = base_url or AI_BASE_URL or DEFAULT_GEMINI_BASE_URL

        if not selected_base:
            selected_base = DEFAULT_GEMINI_BASE_URL

        return key, selected_model, selected_base

    # Generic OpenAI-compatible provider.
    return (
        api_key or AI_API_KEY,
        model or AI_MODEL,
        base_url or AI_BASE_URL,
    )


def _create_client(
    provider: str,
    api_key: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
):
    if OpenAI is None:
        raise RuntimeError(
            "The 'openai' package is not installed. "
            "Run: pip install openai"
        )

    key, selected_model, selected_base = _provider_settings(
        provider,
        api_key,
        model,
        base_url,
    )

    if not key:
        raise RuntimeError(
            "AI_API_KEY is missing. Put your API key in the .env file."
        )

    if not selected_model:
        raise RuntimeError(
            "AI_MODEL is missing. Set AI_MODEL in the .env file."
        )

    kwargs: Dict[str, Any] = {"api_key": key}

    if selected_base:
        kwargs["base_url"] = selected_base

    return OpenAI(**kwargs), selected_model, selected_base


def _extract_tool_call(call: Any) -> Tuple[str, str, str]:
    """Support current OpenAI SDK objects without assuming one exact model."""
    function = getattr(call, "function", None)

    name = getattr(function, "name", "") if function else ""
    arguments = getattr(function, "arguments", "{}") if function else "{}"
    call_id = getattr(call, "id", "") or ""

    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)

    return str(name), str(arguments or "{}"), str(call_id)


def _tool_call_to_dict(call: Any) -> Dict[str, Any]:
    if hasattr(call, "model_dump"):
        try:
            return call.model_dump()
        except Exception:
            pass

    name, arguments, call_id = _extract_tool_call(call)

    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }

def Agent_run_arc(
    user_input: str,
    session_history: Optional[List[Dict[str, Any]]] = None,
    max_steps: int = 5,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Main public agent function.

    Returns:
        (response_text, updated_session_history)
    """
    if not isinstance(user_input, str) or not user_input.strip():
        return "Please enter a message.", session_history or []

    if session_history is None:
        session_history = []

    if not isinstance(session_history, list):
        return "Invalid session history.", []

    max_steps = max(1, int(max_steps))

    system_instruction = (
        "You are a helpful AI assistant. "
        "Use tools only when needed. "
        "After receiving a tool result, explain the result clearly. "
        "Never expose raw internal tool-call details unless the user asks. "
        "If a tool fails, explain the failure and continue when possible. "
        "Do not invent tool results."
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_instruction}
    ]

    # Keep only valid OpenAI-style history entries.
    for item in session_history:
        if (
            isinstance(item, dict)
            and item.get("role") in {"user", "assistant"}
            and isinstance(item.get("content", ""), str)
        ):
            messages.append({
                "role": item["role"],
                "content": item["content"],
            })

    messages.append({
        "role": "user",
        "content": user_input.strip(),
    })

    primary_provider = AI_PROVIDER
    primary_key = api_key or AI_API_KEY
    primary_model = model or AI_MODEL
    primary_base = base_url or AI_BASE_URL

    providers: List[Tuple[str, str, str, str]] = [
        (
            primary_provider,
            primary_key,
            primary_model,
            primary_base,
        )
    ]

    if (
        AI_FALLBACK_PROVIDER
        and AI_FALLBACK_PROVIDER != primary_provider
        and AI_FALLBACK_API_KEY
        and AI_FALLBACK_MODEL
    ):
        providers.append(
            (
                AI_FALLBACK_PROVIDER,
                AI_FALLBACK_API_KEY,
                AI_FALLBACK_MODEL,
                AI_FALLBACK_BASE_URL,
            )
        )

    last_error = ""

    for provider_index, (
        provider,
        provider_key,
        provider_model,
        provider_base,
    ) in enumerate(providers):
        try:
            client, selected_model, selected_base = _create_client(
                provider,
                provider_key,
                provider_model,
                provider_base,
            )

            for step in range(max_steps):
                try:
                    response = client.chat.completions.create(
                        model=selected_model,
                        messages=messages,
                        tools=TOOL_SCHEMAS,
                        tool_choice="auto",
                    )
                except Exception as exc:
                    last_error = str(exc)

                    # Try fallback provider for the complete request.
                    if provider_index + 1 < len(providers):
                        break

                    return (
                        "I could not reach the AI provider. "
                        f"Provider: {provider}. Error: {exc}",
                        session_history,
                    )

                if not response.choices:
                    last_error = "The AI provider returned no choices."
                    break

                choice = response.choices[0].message
                tool_calls = getattr(choice, "tool_calls", None)

                if tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": getattr(choice, "content", None) or "",
                        "tool_calls": [
                            _tool_call_to_dict(call)
                            for call in tool_calls
                        ],
                    })

                    for call in tool_calls:
                        tool_name, raw_arguments, call_id = (
                            _extract_tool_call(call)
                        )

                        if not tool_name:
                            tool_output = _error(
                                "The model returned a tool call without a name."
                            )
                        else:
                            try:
                                arguments = json.loads(raw_arguments)
                                if not isinstance(arguments, dict):
                                    raise ValueError(
                                        "Tool arguments must be a JSON object."
                                    )
                            except Exception as exc:
                                tool_output = _error(
                                    f"Invalid arguments for '{tool_name}': {exc}"
                                )
                            else:
                                tool = dispatcher_registry.get(tool_name)

                                if tool is None:
                                    tool_output = _error(
                                        f"Unknown tool '{tool_name}'."
                                    )
                                else:
                                    try:
                                        tool_output = tool(**arguments)
                                    except Exception as exc:
                                        tool_output = _error(
                                            f"Tool '{tool_name}' failed: {exc}"
                                        )

                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": str(tool_output),
                        })

                    continue

                final_text = getattr(choice, "content", None) or ""

                if not final_text.strip():
                    final_text = (
                        "The AI provider completed the request but "
                        "returned an empty response."
                    )

                updated_history = list(session_history)
                updated_history.append({
                    "role": "user",
                    "content": user_input.strip(),
                })
                updated_history.append({
                    "role": "assistant",
                    "content": final_text,
                })

                return final_text, updated_history

            # If this provider failed and a fallback exists, continue.
            if provider_index + 1 < len(providers):
                continue

        except Exception as exc:
            last_error = str(exc)
            if provider_index + 1 < len(providers):
                continue

    return (
        "The AI agent could not complete the request. "
        f"Last error: {last_error or 'unknown error'}",
        session_history,
    )


# ---------------------------------------------------------------------------
# Optional convenience alias
# ---------------------------------------------------------------------------

run_agent = Agent_run_arc


if __name__ == "__main__":
    print("Agent.py loaded successfully.")
    print(f"Provider: {AI_PROVIDER}")
    print(f"Model: {AI_MODEL or '(not configured)'}")
    print(f"Registered tools: {', '.join(dispatcher_registry)}")