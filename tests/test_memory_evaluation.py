"""Frozen deterministic gate for the memory foundation."""

import json
from pathlib import Path
import tempfile
import unittest

from tools.evaluate_memory import DEFAULT_SPEC, evaluate_memory, main


class MemoryEvaluationTests(unittest.TestCase):
    def test_frozen_memory_evaluation_passes_all_required_checks(self):
        report = evaluate_memory()
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["failed_checks"], [])
        self.assertTrue(all(report["checks"].values()))
        self.assertLessEqual(report["latency"]["milliseconds_p95"], 50.0)

    def test_changed_required_check_set_fails_closed(self):
        spec = json.loads(DEFAULT_SPEC.read_text())
        spec["required_checks"].remove("deletion")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "changed.json"
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "required_checks"):
                evaluate_memory(path)


if __name__ == "__main__":
    unittest.main()
