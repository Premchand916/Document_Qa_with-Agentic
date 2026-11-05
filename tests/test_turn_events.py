"""Append-only, owner-scoped turn events and cancellation state."""

from types import SimpleNamespace
from contextlib import closing
import sqlite3
import threading
from unittest.mock import patch

from tests.support import OfflineCase


class TurnEventTests(OfflineCase):
    def repositories(self):
        from runtime.events.repository import TurnEventRepository
        from storage.conversations import ConversationRepository
        root = self.temp / "events"
        return ConversationRepository(root, lease_seconds=10), TurnEventRepository(root)

    def test_lifecycle_events_replay_in_order_after_restart(self):
        from runtime.events.repository import TurnEventRepository
        conversations, events = self.repositories()
        turn = conversations.reserve("alpha", "alice", None, "turn-one", "Alpha?", now=100)
        conversations.complete("alpha", "alice", turn.conversation_id, "turn-one",
                               turn.lease_token,
                               {"response": "30 days", "result_status": "answered",
                                "corpus_version": "v1"}, now=101)
        replay = TurnEventRepository(events.root).replay(
            "alpha", "alice", turn.conversation_id)
        self.assertEqual([event["event_type"] for event in replay],
                         ["turn.started", "message.completed", "turn.completed"])
        self.assertEqual([event["sequence"] for event in replay], [1, 2, 3])
        self.assertEqual(replay[1]["payload"]["response"], "30 days")
        self.assertNotIn("workspace_id", replay[0])
        self.assertNotIn("principal_id", replay[0])

    def test_append_is_idempotent_and_replay_has_a_cursor(self):
        from domain.events import EventType
        from runtime.events.repository import EventConflict
        conversations, events = self.repositories()
        turn = conversations.reserve("alpha", "alice", None, "turn-one", "Alpha?", now=100)
        first = events.append("alpha", "alice", turn.conversation_id, "turn-one",
                              EventType.TOOL_STARTED, {"tool": "documents.search"},
                              dedupe_key="tool:1", now=101)
        repeated = events.append("alpha", "alice", turn.conversation_id, "turn-one",
                                 EventType.TOOL_STARTED, {"tool": "documents.search"},
                                 dedupe_key="tool:1", now=999)
        self.assertEqual(first.event_id, repeated.event_id)
        tail = events.replay("alpha", "alice", turn.conversation_id,
                             after_sequence=first.sequence - 1)
        self.assertEqual([event["event_type"] for event in tail], ["tool.started"])
        with self.assertRaises(EventConflict):
            events.append("alpha", "alice", turn.conversation_id, "turn-one",
                          EventType.TOOL_FAILED, {"tool": "documents.search"},
                          dedupe_key="tool:1")

    def test_event_stream_is_owner_scoped_and_rejects_secret_fields(self):
        from domain.events import EventType
        from runtime.events.repository import EventStreamNotFound
        conversations, events = self.repositories()
        turn = conversations.reserve("alpha", "alice", None, "turn-one", "Alpha?")
        for workspace, principal in (("alpha", "bob"), ("beta", "alice")):
            with self.assertRaises(EventStreamNotFound):
                events.replay(workspace, principal, turn.conversation_id)
        with self.assertRaises(ValueError):
            events.append("alpha", "alice", turn.conversation_id, "turn-one",
                          EventType.TOOL_COMPLETED, {"api_key": "forbidden"},
                          dedupe_key="secret")

    def test_cancellation_request_is_idempotent_durable_and_visible(self):
        from domain.events import EventType
        from runtime.events.repository import EventConflict, TurnEventRepository
        conversations, events = self.repositories()
        turn = conversations.reserve("alpha", "alice", None, "turn-one", "Alpha?", now=100)
        first = events.request_cancel("alpha", "alice", turn.conversation_id, "turn-one", now=101)
        repeated = events.request_cancel("alpha", "alice", turn.conversation_id, "turn-one", now=102)
        self.assertEqual(first["requested_at"], 101)
        self.assertEqual(repeated["requested_at"], 101)
        reopened = TurnEventRepository(events.root)
        self.assertTrue(reopened.cancellation(
            "alpha", "alice", turn.conversation_id, "turn-one")["cancel_requested"])
        replay = reopened.replay("alpha", "alice", turn.conversation_id)
        self.assertEqual([event["event_type"] for event in replay],
                         ["turn.started", "turn.cancel_requested", "turn.cancelled"])
        self.assertEqual(first["cancelled_at"], 101)
        self.assertEqual(conversations.turn_status("alpha", "alice", turn.conversation_id,
                                                   "turn-one"), "cancelled")
        from storage.conversations import ConversationConflict
        with self.assertRaises(ConversationConflict):
            conversations.complete("alpha", "alice", turn.conversation_id, "turn-one",
                                   turn.lease_token, {"response": "finished"}, now=103)
        self.assertEqual(conversations.get("alpha", "alice", turn.conversation_id)["messages"][-1]
                         ["content"], "Stopped.")
        self.assertEqual(conversations.reserve("alpha", "alice", turn.conversation_id,
                                               "turn-one", "Alpha?").cached_response["result_status"],
                         "cancelled")
        with self.assertRaises(EventConflict):
            reopened.append("alpha", "alice", turn.conversation_id, "turn-one",
                            EventType.TURN_PROGRESS,
                            {"stage": "late"}, dedupe_key="late")

    def test_expired_reclaim_records_failure_before_new_attempt(self):
        conversations, events = self.repositories()
        first = conversations.reserve("alpha", "alice", None, "turn-one", "Alpha?", now=100)
        retry = conversations.reserve("alpha", "alice", first.conversation_id,
                                      "turn-one", "Alpha?", now=111)
        self.assertNotEqual(first.lease_token, retry.lease_token)
        self.assertEqual(retry.event_cursor, 2)
        replay = events.replay("alpha", "alice", first.conversation_id)
        self.assertEqual([event["event_type"] for event in replay],
                         ["turn.started", "turn.failed", "turn.started"])
        self.assertEqual([event["event_type"] for event in events.replay(
            "alpha", "alice", first.conversation_id, after_sequence=retry.event_cursor)],
            ["turn.started"])

    def test_delete_cascades_events_and_cancellation_state(self):
        from runtime.events.repository import EventStreamNotFound
        conversations, events = self.repositories()
        turn = conversations.reserve("alpha", "alice", None, "turn-one", "Alpha?")
        events.request_cancel("alpha", "alice", turn.conversation_id, "turn-one")
        conversations.delete("alpha", "alice", turn.conversation_id)
        with self.assertRaises(EventStreamNotFound):
            events.replay("alpha", "alice", turn.conversation_id)

    def test_api_replays_completed_turn_events(self):
        self.seed_workspace()
        self.server._graph = SimpleNamespace(invoke=lambda state: {
            "response": "Alpha deadline: 30 days", "result_status": "answered",
        })
        client = self.client()
        chat = client.post("/api/chat", json={
            "query": "What is the alpha deadline?", "idempotency_key": "event-api-turn",
        })
        self.assertEqual(chat.status_code, 200, chat.text)
        conversation_id = chat.json()["conversation_id"]
        response = client.get(f"/api/conversations/{conversation_id}/events")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual([event["event_type"] for event in payload["events"]],
                         ["turn.started", "message.completed", "turn.completed"])
        cursor = payload["next_sequence"]
        self.assertEqual(client.get(
            f"/api/conversations/{conversation_id}/events?after_sequence={cursor}"
        ).json()["events"], [])
        self.assertEqual(client.get("/api/conversations/not-an-id/events").status_code, 404)

    def test_sse_replays_from_last_event_id_without_duplicates(self):
        self.seed_workspace()
        self.server._graph = SimpleNamespace(invoke=lambda state: {
            "response": "Alpha deadline: 30 days", "result_status": "answered",
        })
        client = self.client()
        chat = client.post("/api/chat", json={"query": "Alpha?", "idempotency_key": "sse-turn"})
        self.assertEqual(chat.status_code, 200, chat.text)
        identifier = chat.json()["conversation_id"]
        path = f"/api/conversations/{identifier}/events/stream?turn_id=sse-turn"
        first = client.get(path)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertIn("event: turn.started", first.text)
        self.assertIn("event: turn.completed", first.text)
        self.assertNotIn("workspace_id", first.text)
        replay = client.get(path, headers={"Last-Event-ID": "2"})
        self.assertEqual(replay.text.count("event: turn.completed"), 1)
        self.assertNotIn("event: message.completed", replay.text)
        self.assertEqual(client.get(path, headers={"Last-Event-ID": "bad"}).status_code, 400)
        self.assertEqual(client.get(path.replace("sse-turn", "unknown")).status_code, 404)

    def test_legacy_turn_schema_migrates_without_losing_events(self):
        from storage.conversations import ConversationRepository
        root = self.temp / "legacy-turns"
        root.mkdir()
        with closing(sqlite3.connect(root / "conversations.sqlite3")) as connection:
            connection.executescript("""
                CREATE TABLE conversations(workspace_id TEXT,principal_id TEXT,
                  conversation_id TEXT,title TEXT,created_at TEXT,updated_at TEXT,
                  PRIMARY KEY(workspace_id,principal_id,conversation_id));
                CREATE TABLE turns(workspace_id TEXT,principal_id TEXT,conversation_id TEXT,
                  sequence INTEGER,idempotency_key TEXT,question TEXT,
                  status TEXT CHECK(status IN ('running','complete','failed')),
                  response_json TEXT,corpus_version TEXT,lease_token TEXT,lease_expires_at INTEGER,
                  PRIMARY KEY(workspace_id,principal_id,conversation_id,sequence),
                  UNIQUE(workspace_id,principal_id,conversation_id,idempotency_key),
                  FOREIGN KEY(workspace_id,principal_id,conversation_id)
                  REFERENCES conversations(workspace_id,principal_id,conversation_id));
                CREATE TABLE turn_events(workspace_id TEXT,principal_id TEXT,
                  conversation_id TEXT,turn_id TEXT,sequence INTEGER,event_id TEXT,
                  dedupe_key TEXT,event_type TEXT,payload_json TEXT,created_at INTEGER,
                  PRIMARY KEY(workspace_id,principal_id,conversation_id,sequence),
                  UNIQUE(workspace_id,principal_id,conversation_id,turn_id,event_id),
                  UNIQUE(workspace_id,principal_id,conversation_id,turn_id,dedupe_key),
                  FOREIGN KEY(workspace_id,principal_id,conversation_id,turn_id)
                  REFERENCES turns(workspace_id,principal_id,conversation_id,idempotency_key)
                  ON DELETE CASCADE);
            """)
            identifier = "a" * 32
            connection.execute("INSERT INTO conversations VALUES (?,?,?,?,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)",
                               ("alpha", "alice", identifier, "Alpha"))
            connection.execute("INSERT INTO turns VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                               ("alpha", "alice", identifier, 1, "legacy", "Alpha?", "failed",
                                None, None, None, None))
            connection.execute("INSERT INTO turn_events VALUES (?,?,?,?,?,?,?,?,?,?)",
                               ("alpha", "alice", identifier, "legacy", 1, "event-1",
                                "turn.started:old", "turn.started", '{"status":"running"}', 100))
            connection.commit()
        repository = ConversationRepository(root)
        self.assertEqual(repository.turn_status("alpha", "alice", identifier, "legacy"), "failed")
        from runtime.events.repository import TurnEventRepository
        self.assertEqual(len(TurnEventRepository(root).replay("alpha", "alice", identifier)), 1)
        with closing(sqlite3.connect(root / "conversations.sqlite3")) as connection:
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            self.assertIn("'cancelled'", connection.execute(
                "SELECT sql FROM sqlite_master WHERE name='turns'").fetchone()[0])
        retried = repository.reserve("alpha", "alice", identifier, "legacy", "Alpha?")
        self.assertIsNotNone(retried.lease_token)

    def test_async_stop_wins_over_late_graph_result(self):
        from fastapi.testclient import TestClient
        from runtime.events.repository import TurnEventRepository
        self.seed_workspace()
        started, release = threading.Event(), threading.Event()

        def slow_graph(state):
            started.set()
            release.wait(timeout=5)
            return {"response": "Late answer", "result_status": "answered"}

        graph = SimpleNamespace(invoke=slow_graph)
        try:
            with patch.object(self.server, "build_graph", return_value=graph):
                with TestClient(self.server.app, raise_server_exceptions=False,
                                client=("127.0.0.1", 12345), base_url="http://127.0.0.1") as client:
                    response = client.post("/api/chat/turns", json={
                        "query": "Alpha?", "idempotency_key": "slow-turn"})
                    self.assertEqual(response.status_code, 202, response.text)
                    identifier = response.json()["conversation_id"]
                    self.assertTrue(started.wait(timeout=5))
                    stopped = client.delete(f"/api/conversations/{identifier}/turns/slow-turn")
                    self.assertEqual(stopped.status_code, 200, stopped.text)
                    self.assertEqual(stopped.json()["cancelled_at"], stopped.json()["requested_at"])
                    release.set()
                    events = TurnEventRepository(self.temp / "vector_db")
                    path = f"/api/conversations/{identifier}/events/stream?turn_id=slow-turn"
                    replay = client.get(path)
                    self.assertEqual(replay.status_code, 200, replay.text)
                    self.assertIn("event: turn.progress", replay.text)
                    self.assertIn("event: turn.cancelled", replay.text)
                    self.assertNotIn("event: turn.completed", replay.text)
                    self.assertEqual(client.get(f"/api/conversations/{identifier}").json()
                                     ["messages"][-1]["content"], "Stopped.")
                    self.assertEqual([item["event_type"] for item in events.replay(
                        "local", "local-developer", identifier)][-1], "turn.cancelled")
        finally:
            release.set()

    def test_async_turn_streams_progress_and_validated_completion(self):
        from fastapi.testclient import TestClient
        self.seed_workspace()
        graph = SimpleNamespace(invoke=lambda state: {
            "response": "Alpha deadline: 30 days", "result_status": "answered",
        })
        with patch.object(self.server, "build_graph", return_value=graph):
            with TestClient(self.server.app, raise_server_exceptions=False,
                            client=("127.0.0.1", 12345), base_url="http://127.0.0.1") as client:
                accepted = client.post("/api/chat/turns", json={
                    "query": "Alpha?", "idempotency_key": "async-complete"})
                self.assertEqual(accepted.status_code, 202, accepted.text)
                identifier = accepted.json()["conversation_id"]
                self.assertEqual(accepted.json()["after_sequence"], 0)
                stream = client.get(f"/api/conversations/{identifier}/events/stream?turn_id=async-complete")
                self.assertEqual(stream.status_code, 200, stream.text)
                self.assertIn("event: turn.progress", stream.text)
                self.assertIn("event: message.completed", stream.text)
                self.assertIn("event: turn.completed", stream.text)
                detail = client.get(f"/api/conversations/{identifier}").json()
                self.assertEqual(detail["messages"][-1]["content"], "Alpha deadline: 30 days")
                cached = client.post("/api/chat/turns", json={
                    "query": "Alpha?", "conversation_id": identifier,
                    "idempotency_key": "async-complete"})
                self.assertEqual(cached.status_code, 200, cached.text)
                self.assertEqual(cached.json()["status"], "complete")


if __name__ == "__main__":
    import unittest
    unittest.main()
