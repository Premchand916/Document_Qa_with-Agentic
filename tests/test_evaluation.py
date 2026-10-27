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

