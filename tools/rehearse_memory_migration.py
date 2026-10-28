"""Export and verify memory migration contract on an isolated SQLite backup."""

import argparse
import json
from pathlib import Path

from storage.memory_portability import (export_memory_snapshot, rehearse_memory_migration,
                                        write_memory_snapshot)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--snapshot-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()
    if args.snapshot_output:
        write_memory_snapshot(export_memory_snapshot(args.data_dir), args.snapshot_output)
    report = rehearse_memory_migration(args.data_dir)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report_output:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if report["passed"] and not report["source_mutated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
