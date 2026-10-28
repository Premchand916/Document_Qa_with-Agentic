"""Generate reproducible copy-only rollback evidence from a two-version corpus."""

import argparse
import json
from pathlib import Path
import tempfile

from langchain_core.documents import Document

from storage.migration_rehearsal import rehearse_rollback
from storage.workspaces import WorkspaceRepository
from vector_Store.faiss_store import build_vector_store, get_cached_embedding


def generate_evidence() -> dict:
    with tempfile.TemporaryDirectory(prefix="docqa-migration-source-") as folder:
        source = Path(folder) / "source"
        repository = WorkspaceRepository(source, get_cached_embedding, "ollama")
        old_store = build_vector_store([
            Document(page_content="Release policy revision one.", metadata={"source": "policy.txt"})
        ])
        old = repository.publish(
            "release-rehearsal", old_store, [], [], ["txt"],
            [{"name": "policy.txt", "sha256": "1" * 64}], expected_version=None,
        )
        current_store = build_vector_store([
            Document(page_content="Release policy revision two.", metadata={"source": "policy.txt"})
        ])
        repository.publish(
            "release-rehearsal", current_store, [], [], ["txt"],
            [{"name": "policy.txt", "sha256": "2" * 64}], expected_version=old,
        )
        report = rehearse_rollback(source, "release-rehearsal", get_cached_embedding)
    return {**report, "fixture": "synthetic_two_version_production_storage"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = generate_evidence()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "source_mutated": report["source_mutated"]}))
    return 0 if report["passed"] and not report["source_mutated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
