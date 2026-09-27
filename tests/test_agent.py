import json
import os
import sys
import shutil
import unittest
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

import Agent

class TestAgentSuite(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    """Set up a clean temporary environment for testing file operations."""
    cls.test_dir = "./test_env"
    cls.output_dir = "./agent_output"
    os.makedirs(cls.test_dir, exist_ok=True)
    os.makedirs(cls.output_dir, exist_ok=True)

  @classmethod
  def tearDownClass(cls):
    """Clean up temporary test files and output folders after tests run."""
    if os.path.exists(cls.test_dir):
      shutil.rmtree(cls.test_dir)
    if os.path.exists(cls.output_dir):
      shutil.rmtree(cls.output_dir)

  def test_01_file_operations(self):
    """Tests save_file and read_file functions for correct file write/read logic."""
    test_file_path = os.path.join(self.test_dir, "test.txt")
    test_content = "Hello Agent"

    # 1. Test Save File
    save_res = Agent.save_file(
        file_path="test.txt", content=test_content, base_dir=self.test_dir
    )

    # Verify save_file succeeds
    if save_res.startswith("{"):
      save_json = json.loads(save_res)
      self.assertEqual(save_json.get("status"), "success")
    else:
      self.assertIn("success", save_res.lower())

    # 2. Test Read File
    read_res = Agent.read_file(test_file_path)

    # Parse read output depending on return type (JSON or plain string)
    if read_res.startswith("{"):
      read_json = json.loads(read_res)
      content = read_json.get("content")
      if isinstance(content, list):
        extracted_text = content[0] if content else ""
      else:
        extracted_text = str(content)
      self.assertEqual(extracted_text, test_content)
    else:
      self.assertEqual(read_res.strip(), test_content)

  def test_02_list_directory(self):
    """Tests directory listing functionality."""
    res = Agent.list_directory(self.test_dir)
    self.assertIn("test.txt", res)

  def test_03_inspect_dataset(self):
    """Tests dataset loading and shape inspection."""
    sample_csv = os.path.join(self.test_dir, "data.csv")
    with open(sample_csv, "w", encoding="utf-8") as f:
      f.write("col1,col2,label\n1,2,0\n3,4,1\n")

    res = Agent.inspect_dataset(sample_csv)
    self.assertIn("col1", res)
    self.assertIn("label", res)

  def test_04_compute_statistics(self):
    """Tests calculation of statistical summaries."""
    sample_csv = os.path.join(self.test_dir, "data.csv")
    res = Agent.compute_statistics(sample_csv, column_name="label")
    self.assertIn("mean", res.lower())

  def test_05_train_model(self):
    """Tests training XGBoost models on temporary data."""
    sample_csv = os.path.join(self.test_dir, "data.csv")
    res = Agent.train_model(
        file_path=sample_csv,
        target_column="label",
        task_type="classification",
    )

    # Checks case-insensitively for accuracy in string or JSON
    self.assertIn("accuracy", res.lower())

  def test_06_dispatcher_registry(self):
    """Tests that all required functions are registered in the dispatcher."""
    required_tools = [
        "read_file",
        "save_file",
        "list_directory",
        "inspect_dataset",
        "compute_statistics",
        "train_xgboost_model",
        "web_scrape",
    ]
    for tool_name in required_tools:
      self.assertIn(tool_name, Agent.dispatcher_registry)


if __name__ == "__main__":
  unittest.main()