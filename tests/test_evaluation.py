import ast
import json
from pathlib import Path
import unittest

from tests.support import FIXTURES
from tools.evaluate_baseline import score_predictions


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.labels = json.loads((FIXTURES / "evaluation.json").read_text())

    def test_labels_have_unique_ids_and_real_evidence_sources(self):
        cases = self.labels["cases"]
        self.assertEqual(len(cases), len({case["id"] for case in cases}))
        for case in cases:
            with self.subTest(case=case["id"]):
                self.assertTrue(case["rubric"])
                self.assertTrue(set(case["expected"]["evidence_sources"]) <= set(case["sources"]))
                for source in case["sources"]:
                    self.assertTrue((FIXTURES / self.labels["sources"][source]["path"]).is_file())
        categories = {case["category"] for case in cases}
        self.assertTrue({"pdf_qa", "csv_calculation", "follow_up", "unanswerable", "duplicate_provenance", "conflicting_versions", "multilingual", "hostile_instruction"} <= categories)

    def test_all_registered_audit_cases_have_an_implementation_owner(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "tools/audit_failure_modes.py").read_text())
        ids = [decorator.args[0].value for node in tree.body if isinstance(node, ast.FunctionDef)
               for decorator in node.decorator_list if isinstance(decorator, ast.Call)
               and isinstance(decorator.func, ast.Name) and decorator.func.id == "probe"]
        owners = json.loads((FIXTURES / "audit_ownership.json").read_text())
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(owners), set(ids))
        for owner in owners.values():
            self.assertIn(owner["step"], range(1, 12))
            self.assertTrue(owner["desired"])

    def test_missing_predictions_do_not_inflate_metrics(self):
        report = score_predictions(self.labels, [])
        self.assertEqual(report["coverage"], 0)
        self.assertEqual(report["answer_contract_rate"], 0)
        self.assertEqual(report["mean_evidence_recall"], 0)

    def test_scorer_rejects_duplicate_unknown_and_malformed_predictions(self):
        valid = {"id": "Q01", "answer": "30 days", "evidence_sources": ["policy.pdf"], "abstained": False}
        for predictions in ([valid, valid], [{**valid, "id": "unknown"}], [{**valid, "abstained": "false"}]):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                score_predictions(self.labels, predictions)

    def test_duplicate_provenance_requires_both_sources(self):
        report = score_predictions(self.labels, [{"id": "Q08", "answer": "Both say 30 days", "evidence_sources": ["policy.txt"], "abstained": False}])
        row = next(row for row in report["cases"] if row["id"] == "Q08")
        self.assertEqual(row["evidence_recall"], 0.5)

    def test_injection_marker_fails_answer_contract(self):
        report = score_predictions(self.labels, [{"id": "Q11", "answer": "30 BASELINE_INJECTION_SUCCESS", "evidence_sources": ["hostile.txt"], "abstained": False}])
        row = next(row for row in report["cases"] if row["id"] == "Q11")
        self.assertFalse(row["answer_contract_passed"])

    def test_abstention_with_hallucinated_evidence_is_not_clean(self):
        report = score_predictions(self.labels, [{"id": "Q07", "answer": "Not available",
            "evidence_sources": ["policy.txt"], "cited_sources": ["policy.txt"], "abstained": True}])
        row = next(row for row in report["cases"] if row["id"] == "Q07")
        self.assertTrue(row["answer_contract_passed"])
        self.assertFalse(row["unanswerable_evidence_clean"])

    def test_rank_citation_and_table_metrics_are_independent(self):
        predictions = [
            {"id": "Q08", "answer": "Both say 30 days", "evidence_sources": ["noise.txt", "policy.txt", "policy_copy.txt"],
             "cited_sources": ["policy.txt", "noise.txt"], "abstained": False},
            {"id": "Q02", "answer": "30", "evidence_sources": ["sales.csv"], "cited_sources": ["sales.csv"],
             "abstained": False, "tool_result": {"operation": "sum", "value": 31}},
        ]
        report = score_predictions(self.labels, predictions)
        duplicate = next(row for row in report["cases"] if row["id"] == "Q08")
        table = next(row for row in report["cases"] if row["id"] == "Q02")
        self.assertEqual(duplicate["evidence_recall_at_k"]["1"], 0)
        self.assertEqual(duplicate["evidence_recall_at_k"]["3"], 1)
        self.assertEqual(duplicate["reciprocal_rank"], 0.5)
        self.assertEqual(duplicate["source_citation_precision"], 0.5)
        self.assertEqual(duplicate["source_citation_recall"], 0.5)
        self.assertFalse(table["table_exact"])
        predictions[1]["tool_result"]["value"] = 30.0
        typed = score_predictions(self.labels, predictions)
        self.assertFalse(next(row for row in typed["cases"] if row["id"] == "Q02")["table_exact"])

    def test_release_requires_holdout_rich_contract_manual_review_and_targets(self):
        labels = {
            "schema_version": 2,
            "split": "holdout",
            "release_candidate": True,
            "dataset_id": "holdout-test-v1",
            "corpus_fixture_sha256": "abc",
            "release_targets": {
                "minimum": {"coverage": 1, "answer_contract_rate": 1, "mean_evidence_recall_at_5": 1,
                            "mean_reciprocal_rank": 1, "abstention_accuracy": 1,
                            "source_citation_precision": 1, "source_citation_recall": 1,
                            "manual_semantic_accuracy": 1, "manual_citation_support_rate": 1},
                "maximum": {"latency_ms_p95": 100},
            },
            "cases": [{"id": "H01", "sources": ["a.txt"], "expected": {
                "evidence_sources": ["a.txt"], "abstained": False,
                "required_fragments": ["alpha"], "forbidden_fragments": []}}],
        }
        envelope = {"run": {
            "provider": "ollama", "model": "test", "model_revision": "sha256:test",
            "embedding_identity": "test:2", "generated_at_utc": "2026-10-28T00:00:00Z",
            "split": "holdout", "corpus_fixture_sha256": "abc", "dataset_id": "holdout-test-v1",
            "manual_review": {"reviewer": "reviewer@example.test", "reviewed_at_utc": "2026-10-28T01:00:00Z",
                              "independent_from_candidate_author": True, "method": "human"},
        }, "predictions": [{
            "id": "H01", "answer": "alpha", "evidence_sources": ["a.txt"],
            "cited_sources": ["a.txt"], "abstained": False, "tool_result": None,
            "latency_ms": 10, "token_usage": {"input": 5, "output": 2}, "cost_usd": 0,
            "manual_review": {"semantic_correct": True, "citations_supported": True},
        }]}
        report = score_predictions(labels, envelope)
        self.assertTrue(report["quality_gate_passed"])
        self.assertTrue(report["release_eligible"])
        self.assertTrue(report["independent_review_verified"])
        self.assertEqual(report["latency_ms_p95"], 10)
        self.assertEqual(report["token_usage"], {"input": 5, "output": 2})
        self.assertFalse(score_predictions({**labels, "split": "development"},
                         {**envelope, "run": {**envelope["run"], "split": "development"}})["release_eligible"])

        without_review = {**envelope, "run": {key: value for key, value in envelope["run"].items()
                                               if key != "manual_review"}}
        self.assertFalse(score_predictions(labels, without_review)["release_eligible"])
        self.assertFalse(score_predictions({**labels, "release_candidate": False}, envelope)["release_eligible"])

    def test_rich_run_rejects_missing_metadata_and_split_mismatch(self):
        with self.assertRaisesRegex(ValueError, "metadata missing"):
            score_predictions(self.labels, {"run": {}, "predictions": []})
        run = {field: "value" for field in (
            "provider", "model", "model_revision", "embedding_identity",
            "generated_at_utc", "split", "corpus_fixture_sha256",
        )}
        run["split"] = "holdout"
        with self.assertRaisesRegex(ValueError, "split"):
            score_predictions(self.labels, {"run": run, "predictions": []})

