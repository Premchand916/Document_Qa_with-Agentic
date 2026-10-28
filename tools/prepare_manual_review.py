"""Create a complete independent-review worksheet from labels and predictions."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def prepare_review(labels: dict, predictions: dict) -> dict:
    cases = {case["id"]: case for case in labels["cases"]}
    rows = {row["id"]: row for row in predictions["predictions"]}
    if set(rows) != set(cases):
        raise ValueError("Review preparation requires exactly one prediction for every label case")
    return {
        "schema_version": 1,
        "dataset_id": labels["dataset_id"],
        "candidate_run": {
            key: predictions["run"][key]
            for key in ("provider", "model", "model_revision", "generated_at_utc", "corpus_fixture_sha256")
        },
        "review_instructions": [
            "Reviewer must not be the candidate-run author.",
            "Set semantic_correct only after checking the rubric and expected contract.",
            "Set citations_supported only after checking cited sources against retrieved evidence.",
            "Complete reviewer metadata and every null decision; then attach with tools.attach_manual_review.",
        ],
        "review": {
            "reviewer": "",
            "reviewed_at_utc": "",
            "independent_from_candidate_author": None,
            "method": "human",
        },
        "cases": [{
            "id": identifier,
            "category": case.get("category"),
            "query": case["query"],
            "rubric": case["rubric"],
            "expected": case["expected"],
            "candidate": {
                "answer": rows[identifier]["answer"],
                "evidence_sources": rows[identifier]["evidence_sources"],
                "cited_sources": rows[identifier]["cited_sources"],
                "abstained": rows[identifier]["abstained"],
                "tool_result": rows[identifier].get("tool_result"),
            },
            "semantic_correct": None,
            "citations_supported": None,
            "notes": "",
        } for identifier, case in cases.items()],
        "prepared_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    worksheet = prepare_review(
        json.loads(args.labels.read_text(encoding="utf-8")),
        json.loads(args.predictions.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(worksheet, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(worksheet['cases'])} review cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
