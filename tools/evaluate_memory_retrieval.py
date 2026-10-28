"""Run label-blind semantic-memory retrieval, then score a separate label bundle."""

import argparse
import json
from pathlib import Path
import tempfile

from memory.semantic_memory import SemanticMemoryRepository


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "tests/fixtures/memory_retrieval/tasks.json"
DEFAULT_LABELS = ROOT / "tests/fixtures/memory_retrieval/labels.json"


def load_tasks(path: Path) -> dict:
    tasks = json.loads(path.read_text())
    if tasks.get("schema_version") != 1 or tasks.get("split") != "development_blind":
        raise ValueError("Invalid memory retrieval task bundle")
    forbidden = {"expected_key", "label", "answer", "target"}
    if any(forbidden & set(case) for case in tasks.get("cases", [])):
        raise ValueError("Memory retrieval tasks contain label fields")
    return tasks


def run_inference(tasks: dict) -> dict:
    predictions = []
    with tempfile.TemporaryDirectory(prefix="docqa-memory-retrieval-") as folder:
        for case in tasks["cases"]:
            repository = SemanticMemoryRepository(Path(folder) / case["id"])
            for memory in case["memories"]:
                repository.upsert(
                    "evaluation", "runner", memory["category"], memory["key"], memory["value"],
                    now=int(memory["updated_at"]),
                )
            recalled = repository.recall("evaluation", "runner", case["query"], now=400)
            predictions.append({"id": case["id"], "ranked_keys": [item["key"] for item in recalled]})
    return {"schema_version": 1, "dataset_id": tasks["dataset_id"], "predictions": predictions}


def score_predictions(predictions: dict, labels_path: Path = DEFAULT_LABELS) -> dict:
    labels = json.loads(labels_path.read_text())
    if (labels.get("schema_version") != 1
            or labels.get("dataset_id") != predictions.get("dataset_id")):
        raise ValueError("Memory retrieval labels do not match predictions")
    expected = {item["id"]: item["expected_key"] for item in labels["labels"]}
    supplied = {item["id"]: item["ranked_keys"] for item in predictions["predictions"]}
    if set(expected) != set(supplied):
        raise ValueError("Memory retrieval case IDs do not match labels")
    passed_cases = sum(bool(supplied[identifier]) and supplied[identifier][0] == key
                       for identifier, key in expected.items())
    recall_at_1 = passed_cases / len(expected)
    target = float(labels["targets"]["recall_at_1"])
    return {**predictions, "recall_at_1": recall_at_1, "target_recall_at_1": target,
            "passed": recall_at_1 >= target}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = score_predictions(run_inference(load_tasks(args.tasks)), args.labels)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
