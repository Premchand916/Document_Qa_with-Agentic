"""Purge expired working, episodic, and semantic memory for a scheduled job."""

import argparse
import json
from pathlib import Path
import time

from memory.episodic_memory import EpisodicMemoryRepository
from memory.semantic_memory import SemanticMemoryRepository
from storage.conversations import ConversationRepository


def purge_memory_stores(data_dir: Path, *, now: float | None = None,
                        conversation_retention_days: int = 90,
                        episode_retention_days: int = 180,
                        semantic_retention_days: int = 365) -> dict:
    current_time = time.time() if now is None else now
    episodes = EpisodicMemoryRepository(
        data_dir, retention_days=episode_retention_days,
    ).purge_expired(now=int(current_time))
    semantic = SemanticMemoryRepository(
        data_dir, retention_days=semantic_retention_days,
    ).purge_expired(now=int(current_time))
    conversations = ConversationRepository(
        data_dir, retention_days=conversation_retention_days,
    ).purge_expired(now=current_time)
    return {
        "schema_version": 1,
        "purged_at": int(current_time),
        "deleted": {
            "working_conversations": conversations,
            "episodic_memories": episodes,
            "semantic_memories": semantic,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--conversation-retention-days", type=int, default=90)
    parser.add_argument("--episode-retention-days", type=int, default=180)
    parser.add_argument("--semantic-retention-days", type=int, default=365)
    args = parser.parse_args()
    report = purge_memory_stores(
        args.data_dir,
        conversation_retention_days=args.conversation_retention_days,
        episode_retention_days=args.episode_retention_days,
        semantic_retention_days=args.semantic_retention_days,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
