"""Append-only, owner-scoped turn events and cancellation state."""

from types import SimpleNamespace

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
                         ["turn.started", "turn.cancel_requested"])
        conversations.complete("alpha", "alice", turn.conversation_id, "turn-one",
                               turn.lease_token, {"response": "finished"}, now=103)
        with self.assertRaises(EventConflict):
            reopened.request_cancel("alpha", "alice", turn.conversation_id, "turn-one")

    def test_expired_reclaim_records_failure_before_new_attempt(self):
        conversations, events = self.repositories()
        first = conversations.reserve("alpha", "alice", None, "turn-one", "Alpha?", now=100)
        retry = conversations.reserve("alpha", "alice", first.conversation_id,
                                      "turn-one", "Alpha?", now=111)
        self.assertNotEqual(first.lease_token, retry.lease_token)
        replay = events.replay("alpha", "alice", first.conversation_id)
        self.assertEqual([event["event_type"] for event in replay],
                         ["turn.started", "turn.failed", "turn.started"])

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


if __name__ == "__main__":
    import unittest
    unittest.main()
