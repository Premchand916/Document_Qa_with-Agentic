"""Operator memory cleanup remains bounded and covers every durable memory store."""

import time

from tests.support import OfflineCase
from tools.purge_memories import purge_memory_stores


class MemoryOperationsTests(OfflineCase):
    def test_scheduled_purge_target_cleans_all_expired_memory_types(self):
        from memory.episodic_memory import EpisodicMemoryRepository
        from memory.semantic_memory import SemanticMemoryRepository
        from storage.conversations import ConversationRepository

        root = self.temp / "memory-operations"
        current = int(time.time())
        conversations = ConversationRepository(root, retention_days=1)
        turn = conversations.reserve("alpha", "alice", None, "operation-key", "Total?", now=current)
        conversations.complete(
            "alpha", "alice", turn.conversation_id, "operation-key", turn.lease_token,
            {"response": "30", "corpus_version": "v1"}, now=current,
        )
        episodes = EpisodicMemoryRepository(root, retention_days=1)
        episodes.save_feedback(
            "alpha", "alice", turn.conversation_id, "operation-key", "Use 30.", now=current,
        )
        SemanticMemoryRepository(root, retention_days=1).upsert(
            "alpha", "alice", "preference", "Answer style", "Concise", now=current,
        )

        report = purge_memory_stores(
            root, now=current + 2 * 24 * 60 * 60,
            conversation_retention_days=1, episode_retention_days=1,
            semantic_retention_days=1,
        )
        self.assertEqual(report["deleted"], {
            "working_conversations": 1,
            "episodic_memories": 1,
            "semantic_memories": 1,
        })
        self.assertEqual(episodes.list_recent("alpha", "alice"), [])
        self.assertEqual(SemanticMemoryRepository(root).list_recent("alpha", "alice"), [])

