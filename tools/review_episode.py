"""Review one feedback episode through the local operator boundary."""

import argparse
import json
from pathlib import Path

from memory.episodic_memory import EpisodicMemoryRepository


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--principal", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--decision", choices=("verified", "rejected"), required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    result = EpisodicMemoryRepository(args.data_dir).review(
        args.workspace, args.principal, args.episode_id, args.decision,
        args.reviewer, note=args.note,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
