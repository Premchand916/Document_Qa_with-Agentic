"""Step 11 exact-versus-ANN benchmark subprocess tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class IndexBenchmarkTests(unittest.TestCase):
    def run_benchmark(self, *arguments):
        environment = {**os.environ, "OMP_NUM_THREADS": "1"}
        environment.pop("KMP_DUPLICATE_LIB_OK", None)
        return subprocess.run(
            [sys.executable, "-m", "tools.benchmark_faiss_indexes", *arguments],
            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30,
        )

    def test_flat_remains_reference_when_exact_latency_target_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "benchmark.json"
            completed = self.run_benchmark("--vectors", "400", "--dimension", "16",
                                           "--queries", "20", "--k", "5",
                                           "--output", str(output))
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads(output.read_text())
        self.assertEqual(report["flat"]["index"], "IndexFlatL2")
        self.assertTrue(report["flat"]["exact"])
        self.assertGreaterEqual(report["hnsw"]["recall_at_k_vs_flat"], 0.8)
        self.assertEqual(report["recommendation"], "retain_flat")
        self.assertEqual(report["reason"], "exact_flat_meets_latency_target")
        self.assertFalse(report["production_index_changed"])

    def test_invalid_vector_contract_fails_before_benchmark(self):
        completed = self.run_benchmark("--vectors", "10")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("vector_count must be between", completed.stderr)


if __name__ == "__main__":
    unittest.main()
