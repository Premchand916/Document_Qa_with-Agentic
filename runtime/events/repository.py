"""Owner-scoped append-only turn events and durable cancellation requests."""

from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3
import time
from uuid import uuid4

from domain.events import EventType, TurnEvent, validated_event_payload
from storage.workspaces import workspace_key


class EventConflict(Exception):
    """A dedupe key was reused with different event content."""


class EventStreamNotFound(Exception):
    """Conversation or turn is missing outside the authorized owner scope."""


class TurnEventRepository:
    def __init__(self, root: Path):
        self.root = Path(root)

    @contextmanager
    def _connect(self):
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        path = self.root / "conversations.sqlite3"
        path.touch(mode=0o600, exist_ok=True)
        connection = sqlite3.connect(path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=10000")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS turn_events (
                    workspace_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL, turn_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL, event_id TEXT NOT NULL,
                    dedupe_key TEXT NOT NULL, event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL, created_at INTEGER NOT NULL,
                    PRIMARY KEY(workspace_id,principal_id,conversation_id,sequence),
                    UNIQUE(workspace_id,principal_id,conversation_id,turn_id,event_id),
                    UNIQUE(workspace_id,principal_id,conversation_id,turn_id,dedupe_key),
                    FOREIGN KEY(workspace_id,principal_id,conversation_id,turn_id)
                      REFERENCES turns(workspace_id,principal_id,conversation_id,idempotency_key)
                      ON DELETE CASCADE);
                CREATE INDEX IF NOT EXISTS turn_events_replay
                    ON turn_events(workspace_id,principal_id,conversation_id,sequence);
                CREATE TABLE IF NOT EXISTS turn_cancellations (
                    workspace_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL, turn_id TEXT NOT NULL,
                    requested_at INTEGER NOT NULL, cancelled_at INTEGER,
                    PRIMARY KEY(workspace_id,principal_id,conversation_id,turn_id),
                    FOREIGN KEY(workspace_id,principal_id,conversation_id,turn_id)
                      REFERENCES turns(workspace_id,principal_id,conversation_id,idempotency_key)
                      ON DELETE CASCADE);
            """)
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _validate(workspace: str, principal: str, conversation_id: str,
                  turn_id: str | None = None, dedupe_key: str | None = None) -> None:
        workspace_key(workspace)
        if not principal or len(principal) > 256:
            raise ValueError("Invalid principal")
        if not re.fullmatch(r"[a-f0-9]{32}", conversation_id):
            raise ValueError("Invalid conversation identifier")
        if turn_id is not None and not 1 <= len(turn_id) <= 128:
            raise ValueError("Invalid turn identifier")
        if dedupe_key is not None and not 1 <= len(dedupe_key) <= 128:
            raise ValueError("Invalid event dedupe key")

    @staticmethod
    def _turn_exists(connection, workspace: str, principal: str,
                     conversation_id: str, turn_id: str | None = None) -> bool:
        query = """SELECT 1 FROM turns WHERE workspace_id=? AND principal_id=? AND conversation_id=?"""
        values: tuple = (workspace, principal, conversation_id)
        if turn_id is not None:
            query += " AND idempotency_key=?"
            values += (turn_id,)
        return connection.execute(query, values).fetchone() is not None

    @staticmethod
    def _row(row: sqlite3.Row) -> TurnEvent:
        return TurnEvent(
            event_id=row["event_id"], workspace_id=row["workspace_id"],
            principal_id=row["principal_id"], conversation_id=row["conversation_id"],
            turn_id=row["turn_id"], sequence=row["sequence"],
            event_type=EventType(row["event_type"]), payload=json.loads(row["payload_json"]),
            created_at=row["created_at"],
        )

    @classmethod
    def _append(cls, connection, workspace: str, principal: str, conversation_id: str,
                turn_id: str, event_type: EventType, encoded: str, safe_payload: dict,
                dedupe_key: str, created_at: int) -> TurnEvent:
        existing = connection.execute("""SELECT * FROM turn_events
            WHERE workspace_id=? AND principal_id=? AND conversation_id=? AND turn_id=?
              AND dedupe_key=?""",
            (workspace, principal, conversation_id, turn_id, dedupe_key)).fetchone()
        if existing is not None:
            if existing["event_type"] != event_type.value or existing["payload_json"] != encoded:
                raise EventConflict()
            return cls._row(existing)
        sequence = connection.execute("""SELECT COALESCE(MAX(sequence),0)+1 FROM turn_events
            WHERE workspace_id=? AND principal_id=? AND conversation_id=?""",
            (workspace, principal, conversation_id)).fetchone()[0]
        event_id = uuid4().hex
        connection.execute("""INSERT INTO turn_events(
            workspace_id,principal_id,conversation_id,turn_id,sequence,event_id,dedupe_key,
            event_type,payload_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (workspace, principal, conversation_id, turn_id, sequence, event_id, dedupe_key,
             event_type.value, encoded, created_at))
        return TurnEvent(event_id, workspace, principal, conversation_id, turn_id,
                         sequence, event_type, safe_payload, created_at)

    def append(self, workspace: str, principal: str, conversation_id: str, turn_id: str,
               event_type: EventType, payload: dict, *, dedupe_key: str,
               now: int | None = None) -> TurnEvent:
        self._validate(workspace, principal, conversation_id, turn_id, dedupe_key)
        if not isinstance(event_type, EventType):
            raise ValueError("Invalid event type")
        safe_payload = validated_event_payload(payload)
        encoded = json.dumps(safe_payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        created_at = int(time.time() if now is None else now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if not self._turn_exists(connection, workspace, principal, conversation_id, turn_id):
                raise EventStreamNotFound()
            event = self._append(connection, workspace, principal, conversation_id, turn_id,
                                 event_type, encoded, safe_payload, dedupe_key, created_at)
            connection.commit()
            return event

    def replay(self, workspace: str, principal: str, conversation_id: str, *,
               turn_id: str | None = None, after_sequence: int = 0,
               limit: int = 100) -> list[dict]:
        self._validate(workspace, principal, conversation_id, turn_id)
        if after_sequence < 0 or not 1 <= limit <= 500:
            raise ValueError("Invalid event replay bounds")
        with self._connect() as connection:
            if not self._turn_exists(connection, workspace, principal, conversation_id, turn_id):
                raise EventStreamNotFound()
            query = """SELECT * FROM turn_events WHERE workspace_id=? AND principal_id=?
                AND conversation_id=? AND sequence>?"""
            values: tuple = (workspace, principal, conversation_id, after_sequence)
            if turn_id is not None:
                query += " AND turn_id=?"
                values += (turn_id,)
            query += " ORDER BY sequence,event_id LIMIT ?"
            values += (limit,)
            return [self._row(row).as_dict() for row in connection.execute(query, values).fetchall()]

    def request_cancel(self, workspace: str, principal: str, conversation_id: str,
                       turn_id: str, *, now: int | None = None) -> dict:
        self._validate(workspace, principal, conversation_id, turn_id)
        requested_at = int(time.time() if now is None else now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("""SELECT status FROM turns WHERE workspace_id=? AND principal_id=?
                AND conversation_id=? AND idempotency_key=?""",
                (workspace, principal, conversation_id, turn_id)).fetchone()
            if row is None:
                raise EventStreamNotFound()
            if row["status"] != "running":
                raise EventConflict()
            connection.execute("""INSERT INTO turn_cancellations(
                workspace_id,principal_id,conversation_id,turn_id,requested_at,cancelled_at)
                VALUES (?,?,?,?,?,NULL)
                ON CONFLICT(workspace_id,principal_id,conversation_id,turn_id) DO NOTHING""",
                (workspace, principal, conversation_id, turn_id, requested_at))
            payload = {"status": "cancel_requested"}
            encoded = json.dumps(payload, separators=(",", ":"))
            self._append(connection, workspace, principal, conversation_id, turn_id,
                         EventType.TURN_CANCEL_REQUESTED, encoded, payload,
                         "turn.cancel_requested", requested_at)
            stored = connection.execute("""SELECT requested_at,cancelled_at FROM turn_cancellations
                WHERE workspace_id=? AND principal_id=? AND conversation_id=? AND turn_id=?""",
                (workspace, principal, conversation_id, turn_id)).fetchone()
            connection.commit()
        return {"turn_id": turn_id, "cancel_requested": True,
                "requested_at": stored["requested_at"], "cancelled_at": stored["cancelled_at"]}

    def cancellation(self, workspace: str, principal: str, conversation_id: str,
                     turn_id: str) -> dict:
        self._validate(workspace, principal, conversation_id, turn_id)
        with self._connect() as connection:
            if not self._turn_exists(connection, workspace, principal, conversation_id, turn_id):
                raise EventStreamNotFound()
            row = connection.execute("""SELECT requested_at,cancelled_at FROM turn_cancellations
                WHERE workspace_id=? AND principal_id=? AND conversation_id=? AND turn_id=?""",
                (workspace, principal, conversation_id, turn_id)).fetchone()
        return {"turn_id": turn_id, "cancel_requested": row is not None,
                "requested_at": row["requested_at"] if row else None,
                "cancelled_at": row["cancelled_at"] if row else None}
