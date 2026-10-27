"""Score explicitly supplied model predictions against synthetic labels, offline.

This measures evidence recall and simple answer contracts, not semantic correctness.
It never invokes a model. Generate real-model predictions separately and opt in by
providing --predictions; no synthetic model scores are presented as quality results.
"""

import argparse
import json
from pathlib import Path


DEFAULT_LABELS = Path(__file__).resolve().parents[1] / "tests/fixtures/evaluation.json"


def score_predictions(labels: dict, predictions: list[dict]) -> dict:
    cases = {case["id"]: case for case in labels["cases"]}
    supplied = {}
    for prediction in predictions:
        identifier = prediction["id"]
        if identifier not in cases or identifier in supplied:
            raise ValueError(f"Unknown or duplicate prediction ID: {identifier}")
        if not isinstance(prediction.get("answer"), str):
            raise ValueError(f"Missing string answer: {identifier}")
        sources = prediction.get("evidence_sources")
        if not isinstance(sources, list) or not all(isinstance(source, str) for source in sources):
            raise ValueError(f"Missing evidence_sources list: {identifier}")
        if not isinstance(prediction.get("abstained"), bool):
            raise ValueError(f"Missing abstained boolean: {identifier}")
        supplied[identifier] = prediction
    rows = []
    for identifier, case in cases.items():
        prediction = supplied.get(identifier)
        expected = case["expected"]
        needed = set(expected["evidence_sources"])
        found = set(prediction["evidence_sources"]) if prediction else set()
        answer = prediction["answer"].casefold() if prediction else ""
        passed = bool(prediction) and (
            prediction["abstained"] == expected["abstained"]
            and all(fragment.casefold() in answer for fragment in expected["required_fragments"])
            and not any(fragment.casefold() in answer for fragment in expected["forbidden_fragments"])
        )
        rows.append({"id": identifier, "provided": prediction is not None,
                     "answer_contract_passed": passed,
                     "evidence_recall": len(needed & found) / len(needed) if needed else None,
                     "unexpected_sources": sorted(found - set(case["sources"]))})
    recalls = [row["evidence_recall"] for row in rows if row["evidence_recall"] is not None]
    return {"scope": "Offline source-level recall and literal answer contracts; manual semantic review required",
            "total_cases": len(rows), "provided_cases": len(supplied),
            "coverage": len(supplied) / len(rows),
            "answer_contract_rate": sum(row["answer_contract_passed"] for row in rows) / len(rows),
            "mean_evidence_recall": sum(recalls) / len(recalls) if recalls else None,
            "cases": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--predictions", type=Path, required=True, help="JSON list; explicit opt-in, no model calls")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = score_predictions(json.loads(args.labels.read_text()), json.loads(args.predictions.read_text()))
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)
    return 0 if report["coverage"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
