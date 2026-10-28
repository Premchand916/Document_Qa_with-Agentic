"""Frozen holdout, label isolation, usage capture, and review handoff tests."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agents.orchestrator_agent import capture_model_calls, invoke_orchestrator
from tools.attach_manual_review import attach_review
from tools.evaluation_dataset import corpus_fixture_sha256, load_inference_manifest
from tools.prepare_manual_review import prepare_review


FIXTURE = Path(__file__).parent / "fixtures/release_holdout"
CALIBRATION_FIXTURE = Path(__file__).parent / "fixtures/holdout"


class HoldoutEvaluationTests(unittest.TestCase):
    def test_frozen_manifest_is_label_blind_and_matches_label_identity(self):
        tasks_path = FIXTURE / "tasks.json"
        tasks = load_inference_manifest(tasks_path)
        labels = json.loads((FIXTURE / "labels.json").read_text())
        self.assertEqual(corpus_fixture_sha256(tasks_path, tasks), labels["corpus_fixture_sha256"])
        self.assertEqual(tasks["dataset_id"], labels["dataset_id"])
        self.assertEqual({case["id"] for case in tasks["cases"]},
                         {case["id"] for case in labels["cases"]})
        self.assertFalse(labels["release_candidate"])

    def test_exposed_calibration_set_cannot_become_release_evidence(self):
        labels = json.loads((CALIBRATION_FIXTURE / "labels.json").read_text())
        self.assertFalse(labels["release_candidate"])

    def test_inference_manifest_rejects_leaked_labels(self):
        manifest = {
            "schema_version": 1, "dataset_id": "leak", "split": "holdout",
            "sources": {"a.txt": {"path": "a.txt", "sha256": "x"}},
            "cases": [{"id": "H1", "query": "q", "expected": {"answer": "secret"}}],
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tasks.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "label fields"):
                load_inference_manifest(path)

    def test_model_capture_records_tokens_without_prompt_or_response_content(self):
        response = SimpleNamespace(
            content="answer",
            usage_metadata={"input_tokens": 11, "output_tokens": 3},
            response_metadata={"model": "llama3.2:3b"},
        )
        model = SimpleNamespace(invoke=lambda prompt: response)
        with patch("agents.orchestrator_agent._get_ollama_settings",
                   return_value=("llama3.2:3b", "http://ollama")), \
                patch("agents.orchestrator_agent._get_ollama_llm", return_value=model), \
                capture_model_calls() as calls:
            self.assertIs(invoke_orchestrator("private prompt", provider="ollama"), response)
        self.assertEqual(calls[0]["token_usage"], {"input": 11, "output": 3})
        self.assertNotIn("private prompt", json.dumps(calls))
        self.assertNotIn("answer", json.dumps(calls))

    def test_review_handoff_rejects_incomplete_and_attaches_complete_decisions(self):
        labels = {
            "dataset_id": "d1",
            "cases": [{"id": "H1", "category": "qa", "query": "q", "rubric": "r",
                       "expected": {"evidence_sources": ["a"], "abstained": False,
                                    "required_fragments": ["a"], "forbidden_fragments": []}}],
        }
        predictions = {
            "run": {"provider": "ollama", "model": "m", "model_revision": "rev",
                    "generated_at_utc": "2026-10-28T00:00:00Z", "corpus_fixture_sha256": "hash",
                    "dataset_id": "d1"},
            "predictions": [{"id": "H1", "answer": "a", "evidence_sources": ["a"],
                             "cited_sources": ["a"], "abstained": False, "tool_result": None}],
        }
        worksheet = prepare_review(labels, predictions)
        with self.assertRaisesRegex(ValueError, "reviewer"):
            attach_review(predictions, worksheet)
        worksheet["review"] = {"reviewer": "independent@example.test",
                               "reviewed_at_utc": "2026-10-28T01:00:00Z",
                               "independent_from_candidate_author": True, "method": "human"}
        worksheet["cases"][0]["semantic_correct"] = True
        worksheet["cases"][0]["citations_supported"] = True
        attached = attach_review(predictions, worksheet)
        self.assertEqual(attached["predictions"][0]["manual_review"], {
            "semantic_correct": True, "citations_supported": True,
        })
        self.assertTrue(attached["run"]["manual_review"]["independent_from_candidate_author"])


if __name__ == "__main__":
    unittest.main()
