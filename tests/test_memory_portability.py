"""Portable memory contract preserves relationships and never mutates source."""

import json
import os
from tests.support import OfflineCase


class MemoryPortabilityTests(OfflineCase):
    def populated_root(self, name="portable"):
        from memory.episodic_memory import EpisodicMemoryRepository
        from memory.semantic_memory import SemanticMemoryRepository
        from storage.conversations import ConversationRepository
        root = self.temp / name
        conversations = ConversationRepository(root)
        turn = conversations.reserve("alpha", "alice", None, "portable-key", "Total?")
        conversations.complete("alpha", "alice", turn.conversation_id, "portable-key", turn.lease_token,
                               {"response": "30", "corpus_version": "v1"})
        failed = conversations.reserve("alpha", "alice", None, "failed-key", "Failed request")
        conversations.fail("alpha", "alice", failed.conversation_id, "failed-key", failed.lease_token)
        semantic = SemanticMemoryRepository(root)
        semantic.upsert("alpha", "alice", "preference", "Style", "Concise")
        episodes = EpisodicMemoryRepository(root)
        saved = episodes.save_feedback("alpha", "alice", turn.conversation_id, "portable-key", "Use 30.")
        episodes.review("alpha", "alice", saved["episode_id"], "verified", "reviewer")
        return root

    def test_snapshot_validates_relationships_and_copy_matches_without_source_mutation(self):
        from storage.memory_portability import export_memory_snapshot, rehearse_memory_migration
        root = self.populated_root()
        snapshot = export_memory_snapshot(root)
        self.assertTrue(snapshot["valid"])
        self.assertEqual(snapshot["counts"]["semantic_memories"], 1)
        self.assertEqual(snapshot["counts"]["episode_reviews"], 1)
        self.assertIn("failed", {row["status"] for row in snapshot["tables"]["turns"]})
        report = rehearse_memory_migration(root)
        self.assertTrue(report["passed"], report)
        self.assertFalse(report["source_mutated"])

    def test_running_turn_blocks_snapshot_and_output_is_private_and_no_overwrite(self):
        from storage.conversations import ConversationRepository
        from storage.memory_portability import (MemoryMigrationBlocked, export_memory_snapshot,
                                                write_memory_snapshot)
        root = self.populated_root()
        ConversationRepository(root).reserve("alpha", "alice", None, "running-key", "Pending")
        with self.assertRaises(MemoryMigrationBlocked):
            export_memory_snapshot(root)
        clean = self.populated_root("clean")
        snapshot = export_memory_snapshot(clean)
        output = self.temp / "snapshot.json"
        write_memory_snapshot(snapshot, output)
        self.assertEqual(os.stat(output).st_mode & 0o777, 0o600)
        self.assertEqual(json.loads(output.read_text())["sha256"], snapshot["sha256"])
        with self.assertRaises(FileExistsError):
            write_memory_snapshot(snapshot, output)
        snapshot["tables"]["semantic_memories"][0]["memory_value"] = "Tampered"
        with self.assertRaisesRegex(ValueError, "checksum"):
            write_memory_snapshot(snapshot, self.temp / "tampered.json")
