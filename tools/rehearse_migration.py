"""Rehearse workspace rollback and restoration on an isolated storage copy."""
import argparse
import json
from pathlib import Path

from vector_Store.faiss_store import get_cached_embedding
from storage.migration_rehearsal import rehearse_rollback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--copy-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = rehearse_rollback(args.data_dir, args.workspace, get_cached_embedding,
                               copy_root=args.copy_dir)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)
    return 0 if report["passed"] and not report["source_mutated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
