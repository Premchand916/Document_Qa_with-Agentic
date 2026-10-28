"""Durable, authorized conversation behavior without external models."""

import asyncio
import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import patch

from tests.support import OfflineCase


class ConversationTests(OfflineCase):
    def test_existing_conversation_database_adds_lease_columns_without_data_loss(self):
        from storage.conversations import ConversationRepository
        root = self.temp / "legacy-conversations"
        root.mkdir()
        database = sqlite3.connect(root / "conversations.sqlite3")
        database.executescript("""
            CREATE TABLE conversations (
                workspace_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL, title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(workspace_id, principal_id, conversation_id));
            CREATE TABLE turns (
                workspace_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL, sequence INTEGER NOT NULL,
                idempotency_key TEXT NOT NULL, question TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('running','complete','failed')),
                response_json TEXT, corpus_version TEXT,
                PRIMARY KEY(workspace_id, principal_id, conversation_id, sequence),
                UNIQUE(workspace_id, principal_id, conversation_id, idempotency_key),
                FOREIGN KEY(workspace_id, principal_id, conversation_id)
                  REFERENCES conversations(workspace_id, principal_id, conversation_id));
        """)
        database.execute("""INSERT INTO conversations(
            workspace_id,principal_id,conversation_id,title) VALUES ('alpha','alice',?,'Existing')""",
                         ("a" * 32,))
        database.execute("""INSERT INTO turns(
            workspace_id,principal_id,conversation_id,sequence,idempotency_key,question,status,response_json)
            VALUES ('alpha','alice',?,1,'existing-key','Existing?','complete','{"response":"Saved"}')""",
                         ("a" * 32,))
        database.commit()
        database.close()

        repo = ConversationRepository(root)
        self.assertEqual(repo.get("alpha", "alice", "a" * 32)["messages"][-1]["content"], "Saved")
        database = sqlite3.connect(root / "conversations.sqlite3")
        try:
            columns = {row[1] for row in database.execute("PRAGMA table_info(turns)")}
        finally:
            database.close()
        self.assertTrue({"lease_token", "lease_expires_at"}.issubset(columns))

    def test_repository_restarts_and_isolates_principals_and_threads(self):
        from storage.conversations import ConversationNotFound, ConversationRepository
        root = self.temp / "conversations"
        repo = ConversationRepository(root)
        first = repo.reserve("alpha", "alice", None, "first-request", "Alpha policy?")
        repo.complete("alpha", "alice", first.conversation_id, "first-request", first.lease_token,
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
        repo.complete("alpha", "alice", first.conversation_id, "same-request", first.lease_token,
                      {"response": "Answer"})
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
        repo = ConversationRepository(root, lease_seconds=10)
        first = repo.reserve("alpha", "alice", None, "request-key", "Alpha?", now=100)
        restarted = ConversationRepository(root, lease_seconds=10)
        self.assertEqual(restarted.recover_expired(now=105), 0)
        self.assertEqual(restarted.recover_expired(now=111), 1)
        retry = restarted.reserve("alpha", "alice", first.conversation_id, "request-key", "Alpha?", now=111)
        self.assertEqual(retry.history, [])
        restarted.complete("alpha", "alice", first.conversation_id, "request-key", retry.lease_token,
                           {"response": "30 days"}, now=112)
        self.assertEqual(restarted.recover_expired(now=200), 0)

    def test_expired_lease_is_reclaimed_and_stale_worker_cannot_mutate_replacement(self):
        from storage.conversations import ConversationConflict, ConversationRepository
        root = self.temp / "conversations"
        first_worker = ConversationRepository(root, lease_seconds=10)
        second_worker = ConversationRepository(root, lease_seconds=10)
        first = first_worker.reserve("alpha", "alice", None, "request-key", "Alpha?", now=100)
        with self.assertRaises(ConversationConflict):
            second_worker.reserve("alpha", "alice", first.conversation_id,
                                  "other-key", "Concurrent?", now=105)
        first_worker.renew("alpha", "alice", first.conversation_id,
                           "request-key", first.lease_token, now=105)
        self.assertEqual(second_worker.recover_expired(now=111), 0)
        self.assertEqual(second_worker.recover_expired(now=116), 1)
        replacement = second_worker.reserve("alpha", "alice", first.conversation_id,
                                            "request-key", "Alpha?", now=116)
        with self.assertRaises(ConversationConflict):
            first_worker.complete("alpha", "alice", first.conversation_id, "request-key",
                                  first.lease_token, {"response": "stale"}, now=117)
        first_worker.fail("alpha", "alice", first.conversation_id, "request-key", first.lease_token)
        second_worker.complete("alpha", "alice", first.conversation_id, "request-key",
                               replacement.lease_token, {"response": "current"}, now=117)
        self.assertEqual(second_worker.get("alpha", "alice", first.conversation_id)
                         ["messages"][-1]["content"], "current")

    def test_long_conversation_has_bounded_recap_and_cursor_pages(self):
        from storage.conversations import ConversationRepository
        repo = ConversationRepository(self.temp / "conversations")
        conversation_id = None
        for sequence in range(1, 13):
            reservation = repo.reserve("alpha", "alice", conversation_id, f"request-{sequence}",
                                       f"Question {sequence}", now=100 + sequence)
            conversation_id = reservation.conversation_id
            repo.complete("alpha", "alice", conversation_id, f"request-{sequence}",
                          reservation.lease_token, {"response": f"Answer {sequence}"},
                          now=100 + sequence)
        next_turn = repo.reserve("alpha", "alice", conversation_id, "request-13", "Question 13", now=200)
        self.assertEqual(next_turn.history[0]["role"], "summary")
        self.assertIn("Question 4", next_turn.history[0]["content"])
        self.assertEqual(len([message for message in next_turn.history if message["role"] == "user"]), 8)
        from utils.prompt_skill import build_prompted_query
        self.assertIn("earlier-summary:", build_prompted_query("Question 13", {"chat_history": next_turn.history}))
        repo.fail("alpha", "alice", conversation_id, "request-13", next_turn.lease_token)

        latest = repo.get("alpha", "alice", conversation_id, limit=5)
        self.assertEqual(latest["summarized_through_sequence"], 4)
        self.assertEqual([m["sequence"] for m in latest["messages"] if m["role"] == "user"],
                         [8, 9, 10, 11, 12])
        older = repo.get("alpha", "alice", conversation_id, limit=5,
                         before_sequence=latest["next_before_sequence"])
        self.assertEqual([m["sequence"] for m in older["messages"] if m["role"] == "user"],
                         [3, 4, 5, 6, 7])

    def test_delete_and_retention_are_atomic_and_owner_scoped(self):
        from storage.conversations import ConversationNotFound, ConversationRepository
        root = self.temp / "conversations"
        repo = ConversationRepository(root, retention_days=30)

        def completed(principal, key):
            reservation = repo.reserve("alpha", principal, None, key, f"Question {key}")
            repo.complete("alpha", principal, reservation.conversation_id, key,
                          reservation.lease_token, {"response": "Answer"})
            return reservation.conversation_id

        deleted_id = completed("alice", "delete-key")
        retained_id = completed("alice", "old-key")
        bob_id = completed("bob", "bob-key")
        with repo._connect() as connection:
            connection.execute("""INSERT INTO conversation_summaries(
                workspace_id,principal_id,conversation_id,through_sequence,summary)
                VALUES ('alpha','alice',?,1,'recap')""", (deleted_id,))
            connection.commit()
        with self.assertRaises(ConversationNotFound):
            repo.delete("alpha", "alice", bob_id)
        repo.delete("alpha", "alice", deleted_id)
        with self.assertRaises(ConversationNotFound):
            repo.get("alpha", "alice", deleted_id)
        database = sqlite3.connect(root / "conversations.sqlite3")
        try:
            self.assertEqual(database.execute("SELECT COUNT(*) FROM turns WHERE conversation_id=?",
                                              (deleted_id,)).fetchone()[0], 0)
            self.assertEqual(database.execute("SELECT COUNT(*) FROM conversation_summaries WHERE conversation_id=?",
                                              (deleted_id,)).fetchone()[0], 0)
            database.execute("UPDATE conversations SET updated_at='2000-01-01 00:00:00' WHERE conversation_id=?",
                             (retained_id,))
            database.commit()
        finally:
            database.close()
        self.assertEqual(repo.purge_expired(), 1)
        with self.assertRaises(ConversationNotFound):
            repo.get("alpha", "alice", retained_id)
        self.assertEqual(repo.get("alpha", "bob", bob_id)["conversation_id"], bob_id)

    def test_startup_recovers_expired_only_and_keeps_live_worker_lease(self):
        from memory.episodic_memory import EpisodicMemoryRepository
        from storage.conversations import ConversationConflict, ConversationRepository
        repo = ConversationRepository(self.temp / "conversations", lease_seconds=60)
        episodes = EpisodicMemoryRepository(self.temp / "conversations")
        active = repo.reserve("alpha", "alice", None, "active-key", "Alpha?")
        with patch.object(self.server, "get_conversation_repository", return_value=repo), \
                patch.object(self.server, "get_episodic_repository", return_value=episodes), \
                patch.object(self.server, "build_graph", return_value=SimpleNamespace()):
            asyncio.run(self.server._startup())
        with self.assertRaises(ConversationConflict):
            repo.reserve("alpha", "alice", active.conversation_id, "other-key", "Other?")
        repo.fail("alpha", "alice", active.conversation_id, "active-key", active.lease_token)

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

    def test_api_deletes_authorized_conversation(self):
        self.seed_workspace()
        self.server._graph = SimpleNamespace(invoke=lambda state: {"response": "Alpha deadline: 30 days"})
        client = self.client()
        created = client.post("/api/chat", json={"query": "Alpha?", "idempotency_key": "delete-api-key"})
        self.assertEqual(created.status_code, 200, created.text)
        conversation_id = created.json()["conversation_id"]
        deleted = client.delete(f"/api/conversations/{conversation_id}")
        self.assertEqual(deleted.status_code, 200, deleted.text)
        self.assertTrue(deleted.json()["deleted"])
        self.assertEqual(client.get(f"/api/conversations/{conversation_id}").status_code, 404)
        self.assertEqual(client.get("/api/conversations").json()["conversations"], [])

    def test_context_free_reference_clarifies_without_running_graph(self):
        self.seed_workspace()
        invocations = []
        self.server._graph = SimpleNamespace(invoke=lambda state: invocations.append(state) or {"response": "wrong"})
        response = self.client().post("/api/chat", json={
            "query": "What is its deadline?", "idempotency_key": "clarify-key"
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["result_status"], "needs_clarification")
        self.assertEqual(response.json()["retrieval_status"]["reason"], "unresolved_reference")
        self.assertEqual(invocations, [])

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

    def test_chat_deadline_keeps_capacity_reserved_until_native_work_exits(self):
        from app.schemas import ChatRequest

        async def run():
            semaphore = asyncio.Semaphore(1)
            with patch.object(self.server, "_CHAT_SLOTS", semaphore), \
                    patch.object(self.server, "_chat_deadline_seconds", return_value=0.01), \
                    patch.object(self.server, "_run_chat_graph",
                                 side_effect=lambda *args: time.sleep(0.06) or {"response": "late"}):
                with self.assertRaises(self.server.ChatDeadlineError):
                    await self.server._run_bounded_chat(
                        ChatRequest(query="alpha"), {"workspace_id": "local"}, [], "a" * 32, "request-key"
                    )
                self.assertEqual(semaphore._value, 0)
                await asyncio.sleep(0.08)
                self.assertEqual(semaphore._value, 1)

        asyncio.run(run())

    def test_chat_deadline_and_provider_failures_have_distinct_safe_codes(self):
        from agents.orchestrator_agent import LLMConfigurationError
        from fastapi.testclient import TestClient
        self.seed_workspace()
        client = self.stack.enter_context(TestClient(
            self.server.app, raise_server_exceptions=False,
            client=("127.0.0.1", 12345), base_url="http://127.0.0.1",
        ))
        self.server._graph = SimpleNamespace(invoke=lambda state: time.sleep(0.06) or {"response": "late"})
        with patch.object(self.server, "_chat_deadline_seconds", return_value=0.01):
            deadline = client.post("/api/chat", json={"query": "alpha", "idempotency_key": "deadline-key"})
        self.assertEqual(deadline.status_code, 504)
        self.assertEqual(deadline.json()["code"], "chat_deadline")

        self.server._graph = SimpleNamespace(
            invoke=lambda state: (_ for _ in ()).throw(LLMConfigurationError("SECRET provider detail"))
        )
        provider = client.post("/api/chat", json={"query": "alpha", "idempotency_key": "provider-key"})
        self.assertEqual(provider.status_code, 503)
        self.assertEqual(provider.json()["code"], "provider_unavailable")
        self.assertNotIn("SECRET", provider.text)
