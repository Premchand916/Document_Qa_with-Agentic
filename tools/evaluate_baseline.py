"""Score explicit model predictions against versioned offline labels.

No model is invoked. Development fixtures can diagnose regressions but can never
become release evidence. Release eligibility requires a held-out label split,
run metadata, full rich prediction contracts, and all declared targets passing.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import median


DEFAULT_LABELS = Path(__file__).resolve().parents[1] / "tests/fixtures/evaluation.json"
RANK_CUTOFFS = (1, 3, 5)
REQUIRED_RUN_FIELDS = {
    "provider", "model", "model_revision", "embedding_identity",
    "generated_at_utc", "split", "corpus_fixture_sha256",
}


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _unique_strings(value, field: str, identifier: str, *, required: bool) -> list[str] | None:
    if value is None and not required:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"Missing {field} list: {identifier}")
    if len(value) != len(set(value)):
        raise ValueError(f"Duplicate {field}: {identifier}")
    return value


def _matches_expected(actual, expected) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _matches_expected(actual[key], value) for key, value in expected.items()
        )
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=1e-9)
    return actual == expected


def _prediction_input(value) -> tuple[dict | None, list[dict], bool]:
    if isinstance(value, list):
        return None, value, False
    if not isinstance(value, dict) or not isinstance(value.get("predictions"), list):
        raise ValueError("Predictions must be a list or a run envelope with a predictions list")
    run = value.get("run")
    if not isinstance(run, dict):
        raise ValueError("Run envelope requires run metadata")
    missing = sorted(REQUIRED_RUN_FIELDS - set(run))
    if missing:
        raise ValueError(f"Run metadata missing fields: {', '.join(missing)}")
    if not all(isinstance(run[field], str) and run[field].strip() for field in REQUIRED_RUN_FIELDS):
        raise ValueError("Run metadata fields must be non-empty strings")
    return run, value["predictions"], True


def _validate_prediction(prediction: dict, identifier: str, *, rich: bool) -> dict:
    if not isinstance(prediction.get("answer"), str):
        raise ValueError(f"Missing string answer: {identifier}")
    evidence = _unique_strings(prediction.get("evidence_sources"), "evidence_sources", identifier, required=True)
    if not isinstance(prediction.get("abstained"), bool):
        raise ValueError(f"Missing abstained boolean: {identifier}")
    cited = _unique_strings(prediction.get("cited_sources"), "cited_sources", identifier, required=rich)
    latency = prediction.get("latency_ms")
    if latency is not None and (not isinstance(latency, (int, float)) or isinstance(latency, bool) or latency < 0):
        raise ValueError(f"Invalid latency_ms: {identifier}")
    if rich and latency is None:
        raise ValueError(f"Missing latency_ms: {identifier}")
    usage = prediction.get("token_usage")
    if usage is not None:
        if (not isinstance(usage, dict) or set(usage) != {"input", "output"}
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in usage.values())):
            raise ValueError(f"Invalid token_usage: {identifier}")
    elif rich:
        raise ValueError(f"Missing token_usage: {identifier}")
    cost = prediction.get("cost_usd")
    if cost is not None and (not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost < 0):
        raise ValueError(f"Invalid cost_usd: {identifier}")
    if rich and "cost_usd" not in prediction:
        raise ValueError(f"Missing cost_usd: {identifier}")
    manual = prediction.get("manual_review")
    if manual is not None and (not isinstance(manual, dict)
            or set(manual) != {"semantic_correct", "citations_supported"}
            or not all(isinstance(value, bool) for value in manual.values())):
        raise ValueError(f"Invalid manual_review: {identifier}")
    return {**prediction, "evidence_sources": evidence, "cited_sources": cited}


def _target_checks(report: dict, targets: dict) -> dict:
    mapping = {
        "coverage": report["coverage"],
        "answer_contract_rate": report["answer_contract_rate"],
        "mean_evidence_recall_at_5": report["evidence_recall_at_k"]["5"],
        "mean_reciprocal_rank": report["mean_reciprocal_rank"],
        "abstention_accuracy": report["abstention_accuracy"],
        "table_exact_accuracy": report["table_exact_accuracy"],
        "source_citation_precision": report["source_citation_precision"],
        "source_citation_recall": report["source_citation_recall"],
        "unanswerable_evidence_clean_rate": report["unanswerable_evidence_clean_rate"],
        "manual_semantic_accuracy": report["manual_semantic_accuracy"],
        "manual_citation_support_rate": report["manual_citation_support_rate"],
    }
    checks = {}
    for name, minimum in targets.get("minimum", {}).items():
        actual = mapping.get(name)
        checks[name] = {"actual": actual, "target": minimum,
                        "passed": actual is not None and actual >= minimum}
    for name, maximum in targets.get("maximum", {}).items():
        actual = report.get(name)
        checks[name] = {"actual": actual, "target": maximum,
                        "passed": actual is not None and actual <= maximum}
    return checks


def score_predictions(labels: dict, predictions) -> dict:
    if labels.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported evaluation schema_version")
    split = labels.get("split", "development")
    if split not in ("development", "holdout"):
        raise ValueError("Evaluation split must be development or holdout")
    if split == "holdout":
        for field in ("dataset_id", "corpus_fixture_sha256"):
            if not isinstance(labels.get(field), str) or not labels[field].strip():
                raise ValueError(f"Holdout labels require {field}")
    run, prediction_rows, rich = _prediction_input(predictions)
    if run is not None and run["split"] != split:
        raise ValueError("Prediction run split does not match label split")
    if run is not None and split == "holdout":
        if run["corpus_fixture_sha256"] != labels["corpus_fixture_sha256"]:
            raise ValueError("Prediction corpus does not match frozen holdout labels")
        if run.get("dataset_id") != labels["dataset_id"]:
            raise ValueError("Prediction dataset does not match holdout labels")

    cases = {case["id"]: case for case in labels["cases"]}
    if len(cases) != len(labels["cases"]):
        raise ValueError("Duplicate label case ID")
    supplied = {}
    for prediction in prediction_rows:
        if not isinstance(prediction, dict) or not isinstance(prediction.get("id"), str):
            raise ValueError("Prediction requires a string ID")
        identifier = prediction["id"]
        if identifier not in cases or identifier in supplied:
            raise ValueError(f"Unknown or duplicate prediction ID: {identifier}")
        supplied[identifier] = _validate_prediction(prediction, identifier, rich=rich)

    rows = []
    for identifier, case in cases.items():
        prediction = supplied.get(identifier)
        expected = case["expected"]
        needed = set(expected["evidence_sources"])
        found_ordered = prediction["evidence_sources"] if prediction else []
        found = set(found_ordered)
        cited_ordered = prediction.get("cited_sources") if prediction else None
        cited = set(cited_ordered or [])
        answer = prediction["answer"].casefold() if prediction else ""
        answer_passed = bool(prediction) and (
            prediction["abstained"] == expected["abstained"]
            and all(fragment.casefold() in answer for fragment in expected["required_fragments"])
            and not any(fragment.casefold() in answer for fragment in expected["forbidden_fragments"])
        )
        recall_at_k = {
            str(cutoff): (len(needed & set(found_ordered[:cutoff])) / len(needed) if needed else None)
            for cutoff in RANK_CUTOFFS
        }
        relevant_ranks = [rank for rank, source in enumerate(found_ordered, 1) if source in needed]
        reciprocal_rank = (1 / min(relevant_ranks)) if needed and relevant_ranks else (0.0 if needed else None)
        citation_precision = (len(cited & needed) / len(cited)) if cited_ordered else None
        citation_recall = (len(cited & needed) / len(needed)) if cited_ordered is not None and needed else None
        expected_tool = expected.get("tool_result")
        table_exact = None if expected_tool is None else bool(prediction) and _matches_expected(
            prediction.get("tool_result"), expected_tool
        )
        unanswerable_clean = None if needed else bool(prediction) and not found and not cited
        manual = prediction.get("manual_review") if prediction else None
        rows.append({
            "id": identifier,
            "provided": prediction is not None,
            "answer_contract_passed": answer_passed,
            "abstention_correct": bool(prediction) and prediction["abstained"] == expected["abstained"],
            "evidence_recall": len(needed & found) / len(needed) if needed else None,
            "evidence_recall_at_k": recall_at_k,
            "reciprocal_rank": reciprocal_rank,
            "source_citation_precision": citation_precision,
            "source_citation_recall": citation_recall,
            "table_exact": table_exact,
            "unanswerable_evidence_clean": unanswerable_clean,
            "manual_semantic_correct": manual.get("semantic_correct") if manual else None,
            "manual_citations_supported": manual.get("citations_supported") if manual else None,
            "unexpected_sources": sorted(found - set(case["sources"])),
            "unexpected_citations": sorted(cited - set(case["sources"])),
        })

    recalls = [row["evidence_recall"] for row in rows if row["evidence_recall"] is not None]
    ranked = {str(cutoff): [row["evidence_recall_at_k"][str(cutoff)] for row in rows
                            if row["evidence_recall_at_k"][str(cutoff)] is not None]
              for cutoff in RANK_CUTOFFS}
    latencies = [float(prediction["latency_ms"]) for prediction in supplied.values()
                 if prediction.get("latency_ms") is not None]
    table_scores = [row["table_exact"] for row in rows if row["table_exact"] is not None]
    citation_precisions = [row["source_citation_precision"] for row in rows
                           if row["source_citation_precision"] is not None]
    citation_recalls = [row["source_citation_recall"] for row in rows
                        if row["source_citation_recall"] is not None]
    semantic_reviews = [row["manual_semantic_correct"] for row in rows
                        if row["manual_semantic_correct"] is not None]
    citation_reviews = [row["manual_citations_supported"] for row in rows
                        if row["manual_citations_supported"] is not None]
    unanswerable_scores = [row["unanswerable_evidence_clean"] for row in rows
                           if row["unanswerable_evidence_clean"] is not None]
    usage_rows = [prediction["token_usage"] for prediction in supplied.values()
                  if prediction.get("token_usage") is not None]
    cost_rows = [float(prediction["cost_usd"]) for prediction in supplied.values()
                 if prediction.get("cost_usd") is not None]

    report = {
        "scope": "Offline labeled scoring; semantic correctness requires independent manual review",
        "label_split": split,
        "release_dataset": labels.get("release_candidate") is True,
        "run": run,
        "rich_prediction_contract": rich,
        "total_cases": len(rows),
        "provided_cases": len(supplied),
        "coverage": len(supplied) / len(rows),
        "answer_contract_rate": _mean([float(row["answer_contract_passed"]) for row in rows]),
        "abstention_accuracy": _mean([float(row["abstention_correct"]) for row in rows]),
        "mean_evidence_recall": _mean(recalls),
        "evidence_recall_at_k": {key: _mean(values) for key, values in ranked.items()},
        "mean_reciprocal_rank": _mean([row["reciprocal_rank"] for row in rows
                                        if row["reciprocal_rank"] is not None]),
        "source_citation_precision": _mean(citation_precisions),
        "source_citation_recall": _mean(citation_recalls),
        "unanswerable_evidence_clean_rate": _mean([float(value) for value in unanswerable_scores]),
        "table_exact_accuracy": _mean([float(value) for value in table_scores]),
        "manual_semantic_accuracy": _mean([float(value) for value in semantic_reviews]),
        "manual_citation_support_rate": _mean([float(value) for value in citation_reviews]),
        "latency_ms_p50": median(latencies) if latencies else None,
        "latency_ms_p95": _percentile(latencies, 0.95),
        "token_usage": None if not usage_rows else {
            "input": sum(row["input"] for row in usage_rows),
            "output": sum(row["output"] for row in usage_rows),
        },
        "cost_usd": sum(cost_rows) if cost_rows else (0.0 if rich else None),
        "cases": rows,
    }
    checks = _target_checks(report, labels.get("release_targets", {}))
    report["target_checks"] = checks
    report["quality_gate_passed"] = report["coverage"] == 1 and bool(checks) and all(
        check["passed"] for check in checks.values()
    )
    review_metadata = run.get("manual_review") if run else None
    report["independent_review_verified"] = bool(
        isinstance(review_metadata, dict)
        and all(isinstance(review_metadata.get(field), str) and review_metadata[field].strip()
                for field in ("reviewer", "reviewed_at_utc", "method"))
        and review_metadata.get("independent_from_candidate_author") is True
    )
    report["release_eligible"] = bool(
        split == "holdout" and labels.get("release_candidate") is True
        and rich and run and report["quality_gate_passed"]
        and len(semantic_reviews) == len(rows) and len(citation_reviews) == len(rows)
        and report["independent_review_verified"]
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Legacy JSON list or versioned run envelope; never invokes a model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--enforce-targets", action="store_true")
    parser.add_argument("--require-release-eligible", action="store_true")
    args = parser.parse_args()
    report = score_predictions(json.loads(args.labels.read_text()), json.loads(args.predictions.read_text()))
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)
    if args.require_release_eligible:
        return 0 if report["release_eligible"] else 1
    if args.enforce_targets:
        return 0 if report["quality_gate_passed"] else 1
    return 0 if report["coverage"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
