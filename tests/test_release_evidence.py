"""Step 11 combined release-evidence gate tests."""
import unittest

from tools.verify_release_evidence import verify_release_evidence


class ReleaseEvidenceTests(unittest.TestCase):
    def inputs(self):
        baseline = {"stages": [{"exit_code": 0}], "diagnostic_gate_passed": True}
        evaluation = {"release_eligible": True}
        migration = {"passed": True, "source_mutated": False}
        index = {"flat": {"exact": True}, "recommendation": "retain_flat",
                 "production_index_changed": False}
        return baseline, evaluation, migration, index

    def test_complete_evidence_bundle_passes(self):
        report = verify_release_evidence(*self.inputs())
        self.assertTrue(report["release_evidence_complete"])
        self.assertEqual(report["blocking_checks"], [])

    def test_development_or_incomplete_evaluation_blocks_release(self):
        baseline, evaluation, migration, index = self.inputs()
        evaluation["release_eligible"] = False
        report = verify_release_evidence(baseline, evaluation, migration, index)
        self.assertFalse(report["release_evidence_complete"])
        self.assertEqual(report["blocking_checks"], ["heldout_model_evaluation_passed"])


if __name__ == "__main__":
    unittest.main()
