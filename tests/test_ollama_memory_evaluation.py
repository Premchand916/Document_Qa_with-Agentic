"""Paired Ollama memory evaluation keeps labels out of inference."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from tools.run_ollama_memory_evaluation import (load_memory_tasks, run_memory_inference,
                                                memory_task_sha256, score_memory_predictions)


class OllamaMemoryEvaluationTests(unittest.TestCase):
    def test_inference_is_label_blind_and_scoring_is_separate(self):
        tasks = load_memory_tasks(Path("tests/fixtures/memory_model/tasks.json"))
        answers = iter(["30 days", "- 30 days", "30 days", "30 days", "help@example.test", "help@example.test"])
        response = lambda *_args, **_kwargs: SimpleNamespace(content=next(answers))
        with patch("tools.run_ollama_memory_evaluation.invoke_orchestrator", side_effect=response):
            predictions = run_memory_inference(tasks, model="test", base_url="http://unused", revision="rev")
        self.assertNotIn("required_fragments", json.dumps(predictions))
        score = score_memory_predictions(predictions)
        self.assertTrue(score["passed"], score)
        self.assertEqual(score["metrics"]["preference_adherence_rate"], 1.0)
        self.assertEqual(score["metrics"]["evidence_safety_rate"], 1.0)

    def test_task_digest_binds_predictions_to_labels(self):
        tasks = load_memory_tasks(Path("tests/fixtures/memory_model/tasks.json"))
        labels = json.loads(Path("tests/fixtures/memory_model/labels.json").read_text())
        self.assertEqual(memory_task_sha256(tasks), labels["task_fixture_sha256"])
        predictions = {"run": {"dataset_id": tasks["dataset_id"],
                               "task_fixture_sha256": "tampered"}, "predictions": []}
        with self.assertRaisesRegex(ValueError, "digest"):
            score_memory_predictions(predictions)

    def test_candidate_bundle_is_sealed_and_covers_declared_boundaries(self):
        tasks = load_memory_tasks(Path("tests/fixtures/memory_model_candidate/tasks.json"))
        labels = json.loads(Path("tests/fixtures/memory_model_candidate/labels.json").read_text())
        self.assertEqual(tasks["split"], "candidate_blind")
        self.assertEqual(labels["evidence_status"], "exposed_calibration")
        self.assertEqual(memory_task_sha256(tasks), labels["task_fixture_sha256"])
        self.assertEqual(len(tasks["cases"]), 8)
        identifiers = {case["id"] for case in tasks["cases"]}
        self.assertTrue({"spanish-preference", "combined-preferences", "multiple-conflicts",
                         "adversarial-memory", "long-context-conflict"}.issubset(identifiers))

    def test_task_bundle_rejects_label_leakage(self):
        tasks = json.loads(Path("tests/fixtures/memory_model/tasks.json").read_text())
        tasks["cases"][0]["semantic_memory"][0]["required_fragments"] = ["30"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tasks.json"
            path.write_text(json.dumps(tasks))
            with self.assertRaisesRegex(ValueError, "label fields"):
                load_memory_tasks(path)
            malformed = json.loads(Path("tests/fixtures/memory_model/tasks.json").read_text())
            malformed["cases"][0]["id"] = []
            path.write_text(json.dumps(malformed))
            with self.assertRaisesRegex(ValueError, "unique strings"):
                load_memory_tasks(path)


if __name__ == "__main__":
    unittest.main()
