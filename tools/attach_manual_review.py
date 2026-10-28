"""Validate completed independent review and attach it to a candidate run."""

import argparse
import json
from pathlib import Path


def attach_review(predictions: dict, worksheet: dict) -> dict:
    run = predictions.get("run") or {}
    review = worksheet.get("review") or {}
    required_metadata = ("reviewer", "reviewed_at_utc", "method")
    if any(not isinstance(review.get(key), str) or not review[key].strip() for key in required_metadata):
        raise ValueError("Review requires reviewer, reviewed_at_utc, and method")
    if review.get("independent_from_candidate_author") is not True:
        raise ValueError("Review must explicitly confirm reviewer independence")
    if worksheet.get("dataset_id") != run.get("dataset_id"):
        raise ValueError("Review dataset does not match candidate run")
    candidate_run = worksheet.get("candidate_run") or {}
    for key in ("model_revision", "generated_at_utc", "corpus_fixture_sha256"):
        if candidate_run.get(key) != run.get(key):
            raise ValueError(f"Review candidate metadata mismatch: {key}")
    rows = worksheet.get("cases")
    if not isinstance(rows, list):
        raise ValueError("Review requires cases")
    decisions = {}
    for row in rows:
        identifier = row.get("id") if isinstance(row, dict) else None
        if not isinstance(identifier, str) or identifier in decisions:
            raise ValueError("Review case IDs must be unique strings")
        for field in ("semantic_correct", "citations_supported"):
            if not isinstance(row.get(field), bool):
                raise ValueError(f"Review decision is incomplete: {identifier}.{field}")
        decisions[identifier] = {
            "semantic_correct": row["semantic_correct"],
            "citations_supported": row["citations_supported"],
        }
    prediction_ids = {row["id"] for row in predictions["predictions"]}
    if set(decisions) != prediction_ids:
        raise ValueError("Review must cover exactly every prediction")
    return {
        **predictions,
        "run": {**run, "manual_review": review},
        "predictions": [
            {**row, "manual_review": decisions[row["id"]]}
            for row in predictions["predictions"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = attach_review(
        json.loads(args.predictions.read_text(encoding="utf-8")),
        json.loads(args.reviews.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Attached {len(result['predictions'])} independent review decisions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
