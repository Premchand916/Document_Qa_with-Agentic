"""Combine baseline, held-out evaluation, migration, and index evidence into one gate."""
import argparse
import json
from pathlib import Path


def verify_release_evidence(baseline: dict, evaluation: dict,
                            migration: dict, index_benchmark: dict) -> dict:
    stages = baseline.get("stages") or []
    checks = {
        "baseline_stages_passed": bool(stages) and all(stage.get("exit_code") == 0 for stage in stages),
        "historical_diagnostics_passed": baseline.get("diagnostic_gate_passed") is True,
        "heldout_model_evaluation_passed": evaluation.get("release_eligible") is True,
        "migration_rollback_passed": migration.get("passed") is True,
        "migration_source_immutable": migration.get("source_mutated") is False,
        "index_reference_is_exact": (index_benchmark.get("flat") or {}).get("exact") is True,
        "index_decision_recorded": index_benchmark.get("recommendation") in ("retain_flat", "evaluate_hnsw"),
        "index_runtime_not_mutated": index_benchmark.get("production_index_changed") is False,
    }
    missing = [name for name, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "release_evidence_complete": not missing,
        "blocking_checks": missing,
        "interpretation": (
            "Evidence bundle is complete; deployment-specific approval is still required."
            if not missing else "Release remains blocked until every listed evidence check passes."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--migration", type=Path, required=True)
    parser.add_argument("--index-benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_release_evidence(
        json.loads(args.baseline.read_text()),
        json.loads(args.evaluation.read_text()),
        json.loads(args.migration.read_text()),
        json.loads(args.index_benchmark.read_text()),
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)
    return 0 if report["release_evidence_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
