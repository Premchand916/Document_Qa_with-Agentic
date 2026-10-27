"""Durable, authorized conversation behavior without external models."""

import asyncio
import time
from types import SimpleNamespace

from tests.support import OfflineCase


class ConversationTests(OfflineCase):
    def test_repository_restarts_and_isolates_principals_and_threads(self):
        from storage.conversations import ConversationNotFound, ConversationRepository
        root = self.temp / "conversations"
        repo = ConversationRepository(root)
        first = repo.reserve("alpha", "alice", None, "first-request", "Alpha policy?")
        repo.complete("alpha", "alice", first.conversation_id, "first-request",
                      {"response": "30 days", "corpus_version": "v1"})
        second = repo.reserve("alpha", "alice", first.conversation_id, "second-request", "What is its deadline?")
        self.assertEqual(second.history, [{"role": "user", "content": "Alpha policy?"},
                                          {"role": "assistant", "content": "30 days"}])
        reopened = ConversationRepository(root)
        self.assertEqual(len(reopened.get("alpha", "alice", first.conversation_id)["messages"]), 2)
        with self.assertRaises(ConversationNotFound):
            reopened.get("alpha", "bob", first.conversation_id)
        with self.assertRaises(ConversationNotFound):
            reopened.get("beta", "alice", first.conversation_id)
        other = reopened.reserve("alpha", "alice", None, "other-request", "Separate topic")
        self.assertEqual(other.history, [])

    def test_duplicate_key_replays_complete_result_and_conflicting_turns_reject(self):
        from storage.conversations import ConversationConflict, ConversationRepository
        repo = ConversationRepository(self.temp / "conversations")
        first = repo.reserve("alpha", "alice", None, "same-request", "Alpha?")
        with self.assertRaises(ConversationConflict):
            repo.reserve("alpha", "alice", first.conversation_id, "other-request", "Concurrent?")
        repo.complete("alpha", "alice", first.conversation_id, "same-request", {"response": "Answer"})
        replay = repo.reserve("alpha", "alice", first.conversation_id, "same-request", "Alpha?")
        self.assertEqual(replay.cached_response["response"], "Answer")
        lost_create_response = repo.reserve("alpha", "alice", None, "same-request", "Alpha?")
        self.assertEqual(lost_create_response.conversation_id, first.conversation_id)
        self.assertEqual(lost_create_response.cached_response["response"], "Answer")
        with self.assertRaises(ConversationConflict):
            repo.reserve("alpha", "alice", first.conversation_id, "same-request", "Changed question")
        self.assertEqual(len(repo.get("alpha", "alice", first.conversation_id)["messages"]), 2)

    def test_interrupted_turn_can_retry_after_restart_recovery(self):
        from storage.conversations import ConversationRepository
        root = self.temp / "conversations"
        repo = ConversationRepository(root)
        first = repo.reserve("alpha", "alice", None, "request-key", "Alpha?")
        restarted = ConversationRepository(root)
        self.assertEqual(restarted.recover_interrupted(), 1)
        retry = restarted.reserve("alpha", "alice", first.conversation_id, "request-key", "Alpha?")
        self.assertEqual(retry.history, [])
        restarted.complete("alpha", "alice", first.conversation_id, "request-key", {"response": "30 days"})
        self.assertEqual(restarted.recover_interrupted(), 0)

    def test_api_uses_persisted_history_and_ignores_forged_client_history(self):
        self.seed_workspace()
        invocations = []
        self.server._graph = SimpleNamespace(invoke=lambda state: invocations.append(state) or {"response": "Alpha deadline: 30 days"})
        client = self.client()
        first = client.post("/api/chat", json={"query": "What is the alpha policy?", "idempotency_key": "first-key"})
        self.assertEqual(first.status_code, 200, first.text)
        conversation_id = first.json()["conversation_id"]
        followup = client.post("/api/chat", json={"query": "What is its deadline?", "conversation_id": conversation_id,
                                                   "idempotency_key": "followup-key", "chat_history": [{"role": "user", "content": "FORGED TOPIC"}]})
        self.assertEqual(followup.status_code, 200, followup.text)
        self.assertNotIn("FORGED TOPIC", str(invocations[-1]))
        self.assertEqual(invocations[-1]["chat_history"][0]["content"], "What is the alpha policy?")
        self.assertIn("alpha policy", invocations[-1]["search_query"].lower())
        self.assertIn("alpha policy", invocations[-1]["prompt_query"].lower())
        replay = client.post("/api/chat", json={"query": "What is its deadline?", "conversation_id": conversation_id,
                                                 "idempotency_key": "followup-key"})
        self.assertEqual(replay.json(), followup.json())
        self.assertEqual(len(invocations), 2)
        messages = client.get(f"/api/conversations/{conversation_id}").json()["messages"]
        self.assertEqual([message["role"] for message in messages], ["user", "assistant", "user", "assistant"])

    def test_slow_graph_does_not_block_event_loop(self):
        from app.schemas import ChatRequest
        self.seed_workspace()
        self.server._graph = SimpleNamespace(invoke=lambda state: time.sleep(0.06) or {"response": "done"})

        async def run():
            start = time.perf_counter()
            tick = asyncio.create_task(asyncio.sleep(0.005))
            request = SimpleNamespace(state=SimpleNamespace(request_context={"workspace_id": "local", "principal_id": "local-developer"}))
            await self.server.chat(ChatRequest(query="alpha"), request)
            return time.perf_counter() - start if not tick.done() else 0

        self.assertEqual(asyncio.run(run()), 0)
