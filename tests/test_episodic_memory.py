"""Selective episodic feedback remains opt-in, scoped, fresh, and non-evidentiary."""

from types import SimpleNamespace

from tests.support import OfflineCase


class EpisodicMemoryTests(OfflineCase):
    @staticmethod
    def completed_turn(root, *, workspace="alpha", principal="alice", question="Total amount?",
                       request_key="request-key", corpus_version="v1"):
        from storage.conversations import ConversationRepository
        conversations = ConversationRepository(root)
        reservation = conversations.reserve(workspace, principal, None, request_key, question, now=100)
        conversations.complete(
            workspace, principal, reservation.conversation_id, request_key, reservation.lease_token,
            {"response": "Original answer", "corpus_version": corpus_version,
             "documents": [{"page_content": "evidence", "metadata": {"source_id": "source-1"}}],
             "citations": [{"evidence_id": "E1", "corpus_version": corpus_version}],
             "response_plan": {"intent": "data_analysis"},
             "tool_result": {"operation": "sum", "value": 30}, "result_status": "answered"},
            now=101,
        )
        return conversations, reservation

    def test_verified_episode_recall_is_exactly_scoped_and_fresh(self):
        from memory.episodic_memory import EpisodicMemoryRepository
        root = self.temp / "episodes"
        _, turn = self.completed_turn(root)
        episodes = EpisodicMemoryRepository(root, retention_days=10)
        saved = episodes.save_feedback("alpha", "alice", turn.conversation_id, "request-key",
                                       "Use the validated total of 30.", now=200)
        self.assertEqual(saved["validation_status"], "unverified")
        self.assertEqual(episodes.recall("alpha", "alice", "total amount", "v1", now=201), [])
        episodes.set_status("alpha", "alice", saved["episode_id"], "verified")
        recalled = episodes.recall("alpha", "alice", "TOTAL amount!", "v1", now=201)
        self.assertEqual(recalled[0]["correction"], "Use the validated total of 30.")
        self.assertEqual(recalled[0]["source_ids"], ["source-1"])
        self.assertEqual(recalled[0]["tool_context"]["tool_result"]["value"], 30)
        self.assertEqual(episodes.recall("alpha", "bob", "total amount", "v1", now=201), [])
        self.assertEqual(episodes.recall("beta", "alice", "total amount", "v1", now=201), [])
        self.assertEqual(episodes.recall("alpha", "alice", "total amount", "v2", now=201), [])
        self.assertEqual(episodes.recall("alpha", "alice", "different task", "v1", now=201), [])
        self.assertEqual(episodes.recall("alpha", "alice", "total amount", "v1",
                                         now=saved["expires_at"]), [])

    def test_rejected_and_deleted_episodes_never_recall_and_transcript_survives_episode_delete(self):
        from memory.episodic_memory import EpisodeNotFound, EpisodicMemoryRepository
        root = self.temp / "episodes"
        conversations, turn = self.completed_turn(root)
        episodes = EpisodicMemoryRepository(root)
        saved = episodes.save_feedback("alpha", "alice", turn.conversation_id, "request-key",
                                       "Rejected correction", now=200)
        episodes.set_status("alpha", "alice", saved["episode_id"], "rejected")
        self.assertEqual(episodes.recall("alpha", "alice", "Total amount?", "v1", now=201), [])
        episodes.delete("alpha", "alice", saved["episode_id"])
        with self.assertRaises(EpisodeNotFound):
            episodes.delete("alpha", "alice", saved["episode_id"])
        self.assertEqual(conversations.get("alpha", "alice", turn.conversation_id)
                         ["messages"][-1]["content"], "Original answer")

    def test_conversation_delete_cascades_linked_episodes(self):
        from memory.episodic_memory import EpisodicMemoryRepository
        root = self.temp / "episodes"
        conversations, turn = self.completed_turn(root)
        episodes = EpisodicMemoryRepository(root)
        episodes.save_feedback("alpha", "alice", turn.conversation_id, "request-key",
                               "Delete with conversation", now=200)
        self.assertEqual(len(episodes.list_recent("alpha", "alice")), 1)
        conversations.delete("alpha", "alice", turn.conversation_id)
        self.assertEqual(episodes.list_recent("alpha", "alice"), [])

    def test_public_api_cannot_forge_verification_and_verified_feedback_guides_prompt_only(self):
        self.seed_workspace()
        invocations = []
        self.server._graph = SimpleNamespace(
            invoke=lambda state: invocations.append(state) or {"response": "Original answer"}
        )
        client = self.client()
        first = client.post("/api/chat", json={"query": "Total amount?", "idempotency_key": "episode-turn-key"})
        self.assertEqual(first.status_code, 200, first.text)
        payload = {"conversation_id": first.json()["conversation_id"],
                   "idempotency_key": first.json()["idempotency_key"],
                   "correction": "Use the validated total of 30."}
        forged = client.post("/api/episodes", json={**payload, "validation_status": "verified"})
        self.assertEqual(forged.status_code, 422)
        saved = client.post("/api/episodes", json=payload)
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertEqual(saved.json()["validation_status"], "unverified")
        self.assertEqual(client.post("/api/episodes", json=payload).status_code, 409)
        episode_id = saved.json()["episode_id"]
        self.assertEqual(client.get("/api/episodes").json()["episodes"][0]["episode_id"], episode_id)
        self.server.get_episodic_repository().set_status("local", "local-developer", episode_id, "verified")

        second = client.post("/api/chat", json={"query": "Total amount?", "idempotency_key": "episode-repeat-key"})
        self.assertEqual(second.status_code, 200, second.text)
        self.assertIn("Use the validated total of 30.", invocations[-1]["prompt_query"])
        self.assertIn("not evidence", invocations[-1]["prompt_query"])
        self.assertEqual(invocations[-1]["documents"] if "documents" in invocations[-1] else [], [])
        self.assertEqual(client.delete(f"/api/episodes/{episode_id}").status_code, 200)
        self.assertEqual(client.get("/api/episodes").json()["episodes"], [])

