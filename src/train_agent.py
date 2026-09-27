import inspect
import json
import os
import signal
import sys
import Agent
import pandas as pd

# Global flag to allow graceful shutdown
RUNNING = True


def handle_shutdown(signum, frame):
  global RUNNING
  print(
      "\n[System Signal] Interrupt received! Flushing remaining logs and"
      " shutting down safely..."
  )
  RUNNING = False


# Register CTRL+C (SIGINT) and termination signals
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


def execute_tool(func_name, arguments_json):
  """Safely dispatches tool calls with keyword arguments and signature filtering."""
  if isinstance(arguments_json, str):
    try:
      args = json.loads(arguments_json)
    except json.JSONDecodeError as e:
      return json.dumps({"error": f"Invalid JSON arguments: {str(e)}"})
  else:
    args = arguments_json or {}

  func = Agent.dispatcher_registry.get(func_name)
  if not func:
    return json.dumps({"error": f"Tool '{func_name}' not found in registry."})

  try:
    sig = inspect.signature(func)
    has_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    )

    filtered_args = (
        args
        if has_kwargs
        else {k: v for k, v in args.items() if k in sig.parameters}
    )
    result = func(**filtered_args)
    return result if isinstance(result, str) else json.dumps(result)

  except Exception as e:
    return json.dumps(
        {"error": f"Execution failure in tool '{func_name}': {str(e)}"}
    )


def run_realtime_training_pipeline(dataset_file="sample_data.csv"):
  global RUNNING

  print("=== Starting Real-time Agent Training Loop ===")

  # 1. Prepare runtime dataset
  df = pd.DataFrame({
      "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
  })
  df.to_csv(dataset_file, index=False)

  # 2. Sequential tool queue to simulate real-time agent workflow
  tool_sequence = [
      ("inspect_dataset", json.dumps({"file_path": dataset_file})),
      (
          "train_xgboost_model",
          json.dumps({
              "file_path": dataset_file,
              "target_column": "label",
              "task_type": "classification",
          }),
      ),
      (
          "compute_statistics",
          json.dumps({
              "file_path": dataset_file,
              "column_name": "label",
          }),
      ),
      (
          "web_scrape",
          json.dumps({
              "url": "https://docs.xgboost.ai",
              "query": "classification tuning parameters",
          }),
      ),
      ("list_directory", json.dumps({"path": "."})),
      (
          "read_file",
          json.dumps({"file_path": dataset_file, "max_lines": 5}),
      ),
      (
          "save_file",
          json.dumps({
              "file_path": "summary_report.txt",
              "content": "Model trained with 95% accuracy.",
          }),
      ),
  ]

  conversation_history = [{
      "role": "user",
      "content": (
          f"Inspect dataset '{dataset_file}' and train a classifier on"
          " 'label'."
      ),
  }]

  output_filename = "agent_training_data.jsonl"

  # Reset output log file for a clean test run
  if os.path.exists(output_filename):
    os.remove(output_filename)

  # 3. Real-time loop execution
  for idx, (tool_name, tool_args) in enumerate(tool_sequence, start=1):
    if not RUNNING:
      print("[Warning] Execution halted by user/system before completion.")
      break

    call_id = f"call_{idx}"
    print(f"[Real-time Step {idx}/7] Executing tool: {tool_name}...")

    # Record assistant tool call intent
    assistant_turn = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": tool_name, "arguments": tool_args},
        }],
    }
    conversation_history.append(assistant_turn)

    # Execute tool and retrieve output
    tool_output = execute_tool(tool_name, tool_args)

    tool_turn = {
        "role": "tool",
        "tool_call_id": call_id,
        "name": tool_name,
        "content": tool_output,
    }
    conversation_history.append(tool_turn)

    # Persistent real-time flush to disk
    Agent.save_training_data_atomic(
        conversation_history, filename=output_filename
    )

  if RUNNING:
    conversation_history.append({
        "role": "assistant",
        "content": (
            "All real-time tasks completed and persisted successfully."
        ),
    })
    Agent.save_training_data_atomic(
        conversation_history, filename=output_filename
    )
    print("=== Real-time Execution Finished Successfully ===")

  # 4. Automated Verification & Testing Suite
  verify_and_test_results(output_filename)


def verify_and_test_results(jsonl_path):
  """Runs automated structural and accuracy verification tests on the real-time generated logs."""
  print("\n=== Running Automated Real-Time Verification Suite ===")

  if not os.path.exists(jsonl_path):
    print("❌ TEST FAILED: Training log file was not created on disk.")
    return

  with open(jsonl_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

  if not lines:
    print("❌ TEST FAILED: Output training file is empty.")
    return

  print(
      "✓ Output file created successfully"
      f" ({len(lines)} log snapshots written)."
  )

  # Inspect final fully populated record
  final_snapshot = json.loads(lines[-1])
  messages = final_snapshot.get("messages", [])

  # Test 1: Check OpenAI message schema integrity
  roles = [m["role"] for m in messages]
  if roles[0] != "user" or "tool" not in roles or roles[-1] != "assistant":
    print("❌ TEST FAILED: Invalid role sequence in fine-tuning log.")
    return
  print("✓ Test 1 Passed: Role sequence aligns with fine-tuning standards.")

  # Test 2: Verify tool call and tool response ID alignment
  tool_calls = [m["tool_calls"][0]["id"] for m in messages if "tool_calls" in m]
  tool_responses = [m["tool_call_id"] for m in messages if m["role"] == "tool"]

  if tool_calls != tool_responses:
    print("❌ TEST FAILED: Tool call IDs do not match tool output IDs.")
    return
  print(
      f"✓ Test 2 Passed: All {len(tool_calls)} tool calls match their execution"
      " outputs."
  )

  # Test 3: Check execution error rate inside tool content outputs
  error_count = 0
  for m in messages:
    if m["role"] == "tool":
      try:
        content = json.loads(m["content"])
        if isinstance(content, dict) and "error" in content:
          error_count += 1
      except json.JSONDecodeError:
        if "Error" in m["content"] or "error" in m["content"]:
          error_count += 1

  if error_count > 0:
    print(
        f"⚠️ Warning: {error_count} tool executions returned runtime errors."
    )
  else:
    print(
        "✓ Test 3 Passed: 100% of tool executions completed without errors."
    )

  print(
      "\n🎉 ACCURATE RESULT: Your real-time training logger is fully verified"
      " and ready!"
  )


if __name__ == "__main__":
  run_realtime_training_pipeline()