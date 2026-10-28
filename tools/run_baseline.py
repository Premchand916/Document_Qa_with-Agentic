"""Run offline controls and known-failure diagnostics as separate subprocesses."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".baseline-results")
    args = parser.parse_args()
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1",
                   "HF_LOCAL_FILES_ONLY": "true", "GOOGLE_API_KEY": "", "GEMINI_API_KEY": "",
                   "TAVILY_API_KEY": "", "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
                   "LANGSMITH_API_KEY": "", "PYTHON_DOTENV_DISABLED": "1", "DOCQA_AUTH_MODE": "local"}
    environment.pop("KMP_DUPLICATE_LIB_OK", None)
    commands = [
        ("tests", ["-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"]),
        ("diagnostics", ["-m", "tools.audit_failure_modes", "--output", str(folder / "diagnostics.json")]),
        ("ingestion", ["-m", "tools.measure_ingestion", "--output", str(folder / "ingestion.json")]),
        ("index_benchmark", ["-m", "tools.benchmark_faiss_indexes", "--output", str(folder / "index-benchmark.json")]),
        ("memory", ["-m", "tools.evaluate_memory", "--output", str(folder / "memory.json")]),
        ("environment", ["-m", "tools.baseline_environment", "--output-dir", str(folder)]),
    ]
    stages = []
    for name, arguments in commands:
        start = time.monotonic()
        with (folder / f"{name}.log").open("w") as log:
            try:
                completed = subprocess.run([sys.executable, *arguments], cwd=ROOT, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT, timeout=180)
                code = completed.returncode
            except subprocess.TimeoutExpired:
                code = 124
        stage = {"name": name, "exit_code": code, "seconds": round(time.monotonic() - start, 3)}
        if name == "tests":
            match = re.search(r"Ran (\d+) tests? in", (folder / "tests.log").read_text())
            stage["tests_run"] = int(match.group(1)) if match else None
        stages.append(stage)
        print(json.dumps(stage), flush=True)
    audit = json.loads((folder / "diagnostics.json").read_text()) if (folder / "diagnostics.json").exists() and stages[1]["exit_code"] == 0 else None
    ingestion = json.loads((folder / "ingestion.json").read_text()) if (folder / "ingestion.json").exists() and stages[2]["exit_code"] == 0 else None
    index_benchmark = json.loads((folder / "index-benchmark.json").read_text()) if (folder / "index-benchmark.json").exists() and stages[3]["exit_code"] == 0 else None
    memory_evaluation = json.loads((folder / "memory.json").read_text()) if (folder / "memory.json").exists() and stages[4]["exit_code"] == 0 else None
    diagnostic_gate = bool(audit) and audit["summary"] == {
        "reproduced": 0, "not_reproduced": 25, "control_passed": 1,
        "control_failed": 0, "probe_error": 0,
    }
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "stages": stages,
              "diagnostic_summary": audit["summary"] if audit else None,
              "diagnostic_gate_passed": diagnostic_gate,
              "ingestion_measurement": ingestion,
              "index_benchmark": index_benchmark,
              "memory_evaluation": memory_evaluation,
              "fixture_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted((ROOT / "tests/fixtures").rglob("*")) if path.is_file()},
              "interpretation": "Offline foundation checks only. Any reproduced defect, failed control, or probe error fails this gate. No live model quality measured."}
    (folder / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all(stage["exit_code"] == 0 for stage in stages) and diagnostic_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
