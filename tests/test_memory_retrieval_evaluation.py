"""Memory retrieval inference remains label-blind and separately scored."""

import json
from pathlib import Path
import tempfile
import unittest

from tools.evaluate_memory_retrieval import load_tasks, run_inference, score_predictions


class MemoryRetrievalEvaluationTests(unittest.TestCase):
    def test_separate_task_bundle_reaches_frozen_recall_target(self):
        predictions = run_inference(load_tasks(
            Path("tests/fixtures/memory_retrieval/tasks.json"),
        ))
        self.assertNotIn("expected_key", json.dumps(predictions))
        report = score_predictions(predictions)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["recall_at_1"], 1.0)

    def test_task_bundle_rejects_label_leakage(self):
        tasks = json.loads(Path("tests/fixtures/memory_retrieval/tasks.json").read_text())
        tasks["cases"][0]["expected_key"] = "Warranty period"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "leaked.json"
            path.write_text(json.dumps(tasks))
            with self.assertRaisesRegex(ValueError, "label fields"):
                load_tasks(path)


if __name__ == "__main__":
    unittest.main()
