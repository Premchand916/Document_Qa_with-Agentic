"""Frozen evaluation-dataset loading and integrity checks."""

import hashlib
import json
from pathlib import Path


FORBIDDEN_INFERENCE_KEYS = {
    "answer", "expected", "forbidden_fragments", "manual_review",
    "release_targets", "required_fragments", "rubric",
}


def _reject_label_leakage(value, location: str = "manifest") -> None:
    if isinstance(value, dict):
        leaked = sorted(FORBIDDEN_INFERENCE_KEYS.intersection(value))
        if leaked:
            raise ValueError(f"Inference manifest contains label fields at {location}: {', '.join(leaked)}")
        for key, child in value.items():
            _reject_label_leakage(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_label_leakage(child, f"{location}[{index}]")


def load_inference_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("split") != "holdout":
        raise ValueError("Inference manifest must use schema_version 1 and holdout split")
    _reject_label_leakage(manifest)
    sources = manifest.get("sources")
    cases = manifest.get("cases")
    if not isinstance(sources, dict) or not sources or not isinstance(cases, list) or not cases:
        raise ValueError("Inference manifest requires non-empty sources and cases")
    identifiers = [case.get("id") for case in cases if isinstance(case, dict)]
    if len(identifiers) != len(cases) or any(not isinstance(item, str) or not item for item in identifiers):
        raise ValueError("Every inference case requires a string ID")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate inference case ID")
    for case in cases:
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError(f"Inference case requires a query: {case['id']}")
    return manifest


def source_paths(manifest_path: Path, manifest: dict) -> dict[str, Path]:
    resolved = {}
    root = manifest_path.resolve().parent
    for name, descriptor in manifest["sources"].items():
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise ValueError("Invalid source descriptor")
        relative = descriptor.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"Missing source path: {name}")
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or path.name != name:
            raise ValueError(f"Invalid or missing frozen source: {name}")
        expected = descriptor.get("sha256")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError(f"Frozen source digest mismatch: {name}")
        resolved[name] = path
    return resolved


def corpus_fixture_sha256(manifest_path: Path, manifest: dict | None = None) -> str:
    manifest = manifest or load_inference_manifest(manifest_path)
    digest = hashlib.sha256()
    raw_manifest = manifest_path.read_bytes()
    digest.update(len(raw_manifest).to_bytes(8, "big"))
    digest.update(raw_manifest)
    for name, path in sorted(source_paths(manifest_path, manifest).items()):
        raw_name, raw_source = name.encode("utf-8"), path.read_bytes()
        digest.update(len(raw_name).to_bytes(8, "big"))
        digest.update(raw_name)
        digest.update(len(raw_source).to_bytes(8, "big"))
        digest.update(raw_source)
    return digest.hexdigest()
