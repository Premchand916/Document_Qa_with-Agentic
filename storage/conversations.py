"""Workspace- and principal-owned durable chat turns.

Only serializable messages and response values enter this store. Graph objects,
FAISS indexes, DataFrames, and provider clients remain request-scoped.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import time
from uuid import uuid4

from storage.workspaces import workspace_key


class ConversationNotFound(Exception):
    """Conversation is missing or does not belong to the request principal."""


class ConversationConflict(Exception):
    """A turn is already running or an idempotency key was reused differently."""


@dataclass(frozen=True)
class TurnReservation:
    conversation_id: str
    history: list[dict]
    lease_token: str | None = None
    cached_response: dict | None = None


class ConversationRepository:
    HISTORY_TURNS = 8
    HISTORY_CHARACTERS = 16_000
    SUMMARY_CHARACTERS = 6_000
    DETAIL_TURN_LIMIT = 50

    def __init__(self, root: Path, *, lease_seconds: int = 60, retention_days: int = 90):
        self.root = Path(root)
        if not 10 <= lease_seconds <= 3600:
            raise ValueError("Lease duration must be between 10 and 3600 seconds")
        if not 1 <= retention_days <= 3650:
            raise ValueError("Retention must be between 1 and 3650 days")
        self.lease_seconds = lease_seconds
        self.retention_days = retention_days

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
                CREATE TABLE IF NOT EXISTS conversations (
                    workspace_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL, title TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(workspace_id, principal_id, conversation_id));
                CREATE TABLE IF NOT EXISTS turns (
                    workspace_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL, sequence INTEGER NOT NULL,
                    idempotency_key TEXT NOT NULL, question TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('running','complete','failed')),
                    response_json TEXT, corpus_version TEXT,
                    lease_token TEXT, lease_expires_at INTEGER,
                    PRIMARY KEY(workspace_id, principal_id, conversation_id, sequence),
                    UNIQUE(workspace_id, principal_id, conversation_id, idempotency_key),
                    FOREIGN KEY(workspace_id, principal_id, conversation_id)
                      REFERENCES conversations(workspace_id, principal_id, conversation_id));
                CREATE UNIQUE INDEX IF NOT EXISTS turns_principal_request
                    ON turns(workspace_id, principal_id, idempotency_key);
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    workspace_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL, through_sequence INTEGER NOT NULL,
                    summary TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(workspace_id, principal_id, conversation_id),
                    FOREIGN KEY(workspace_id, principal_id, conversation_id)
                      REFERENCES conversations(workspace_id, principal_id, conversation_id));
                CREATE INDEX IF NOT EXISTS conversations_retention
                    ON conversations(updated_at);
            """)
            turn_columns = {row[1] for row in connection.execute("PRAGMA table_info(turns)")}
            if "lease_token" not in turn_columns:
                connection.execute("ALTER TABLE turns ADD COLUMN lease_token TEXT")
            if "lease_expires_at" not in turn_columns:
                connection.execute("ALTER TABLE turns ADD COLUMN lease_expires_at INTEGER")
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _validate(workspace: str, principal: str, conversation_id: str | None = None) -> None:
        workspace_key(workspace)
        if not principal or len(principal) > 256:
            raise ValueError("Invalid principal")
        if conversation_id is not None and not re.fullmatch(r"[a-f0-9]{32}", conversation_id):
            raise ValueError("Invalid conversation identifier")

    @classmethod
    def _history(cls, connection, workspace: str, principal: str, conversation_id: str) -> list[dict]:
        summary_row = connection.execute("""SELECT summary FROM conversation_summaries
            WHERE workspace_id=? AND principal_id=? AND conversation_id=?""",
            (workspace, principal, conversation_id)).fetchone()
        rows = connection.execute("""SELECT question, response_json FROM turns
            WHERE workspace_id=? AND principal_id=? AND conversation_id=? AND status='complete'
            ORDER BY sequence DESC LIMIT ?""",
            (workspace, principal, conversation_id, cls.HISTORY_TURNS)).fetchall()
        messages = []
        for row in reversed(rows):
            answer = json.loads(row["response_json"])["response"]
            messages.extend([{"role": "user", "content": row["question"][:4000]},
                             {"role": "assistant", "content": str(answer)[:4000]}])
        summary = summary_row["summary"] if summary_row is not None else ""
        while messages and sum(len(message["content"]) for message in messages) + len(summary) > cls.HISTORY_CHARACTERS:
            messages = messages[2:]
        available = max(0, cls.HISTORY_CHARACTERS - sum(len(message["content"]) for message in messages))
        if summary and available:
            messages.insert(0, {"role": "summary", "content": summary[-available:]})
        return messages

    @classmethod
    def _refresh_summary(cls, connection, workspace: str, principal: str, conversation_id: str) -> None:
        cutoff = connection.execute("""SELECT sequence FROM turns
            WHERE workspace_id=? AND principal_id=? AND conversation_id=? AND status='complete'
            ORDER BY sequence DESC LIMIT 1 OFFSET ?""",
            (workspace, principal, conversation_id, cls.HISTORY_TURNS)).fetchone()
        if cutoff is None:
            connection.execute("""DELETE FROM conversation_summaries
                WHERE workspace_id=? AND principal_id=? AND conversation_id=?""",
                (workspace, principal, conversation_id))
            return
        existing = connection.execute("""SELECT through_sequence,summary FROM conversation_summaries
            WHERE workspace_id=? AND principal_id=? AND conversation_id=?""",
            (workspace, principal, conversation_id)).fetchone()
        through_sequence = existing["through_sequence"] if existing else 0
        if cutoff["sequence"] <= through_sequence:
            return
        rows = connection.execute("""SELECT sequence,question,response_json FROM turns
            WHERE workspace_id=? AND principal_id=? AND conversation_id=? AND status='complete'
              AND sequence>? AND sequence<=? ORDER BY sequence""",
            (workspace, principal, conversation_id, through_sequence, cutoff["sequence"])).fetchall()
        header = "Earlier conversation recap:\n"
        entries = []
        if existing and existing["summary"].startswith(header):
            entries = [entry for entry in existing["summary"][len(header):].split("\n\n") if entry]
        for row in rows:
            answer = str(json.loads(row["response_json"]).get("response", ""))
            entries.append(f"User: {row['question'][:500]}\nAssistant: {answer[:1000]}")
        while entries and len(header) + len("\n\n".join(entries)) > cls.SUMMARY_CHARACTERS:
            entries.pop(0)
        summary = header + "\n\n".join(entries)
        connection.execute("""INSERT INTO conversation_summaries(
            workspace_id,principal_id,conversation_id,through_sequence,summary,updated_at)
            VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(workspace_id,principal_id,conversation_id) DO UPDATE SET
              through_sequence=excluded.through_sequence,summary=excluded.summary,
              updated_at=CURRENT_TIMESTAMP""",
            (workspace, principal, conversation_id, cutoff["sequence"], summary))

    def reserve(self, workspace: str, principal: str, conversation_id: str | None,
                idempotency_key: str, question: str, *, now: int | None = None) -> TurnReservation:
        self._validate(workspace, principal, conversation_id)
        if not idempotency_key or len(idempotency_key) > 128:
            raise ValueError("Invalid idempotency key")
        current_time = int(time.time() if now is None else now)
        lease_token = uuid4().hex
        lease_expires_at = current_time + self.lease_seconds
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            prior_request = connection.execute("""SELECT conversation_id FROM turns
                WHERE workspace_id=? AND principal_id=? AND idempotency_key=?""",
                (workspace, principal, idempotency_key)).fetchone()
            if prior_request is not None and conversation_id not in (None, prior_request["conversation_id"]):
                raise ConversationConflict()
            identifier = conversation_id or (prior_request["conversation_id"] if prior_request else uuid4().hex)
            row = connection.execute("""SELECT 1 FROM conversations
                WHERE workspace_id=? AND principal_id=? AND conversation_id=?""",
                                     (workspace, principal, identifier)).fetchone()
            if row is None:
                if conversation_id is not None:
                    raise ConversationNotFound()
                connection.execute("""INSERT INTO conversations(workspace_id,principal_id,conversation_id,title)
                    VALUES (?,?,?,?)""", (workspace, principal, identifier, question[:120]))
            existing = connection.execute("""SELECT question,status,response_json,lease_expires_at FROM turns
                WHERE workspace_id=? AND principal_id=? AND conversation_id=? AND idempotency_key=?""",
                                          (workspace, principal, identifier, idempotency_key)).fetchone()
            if existing is not None:
                if existing["question"] != question:
                    raise ConversationConflict()
                if existing["status"] == "complete":
                    response = json.loads(existing["response_json"])
                    connection.commit()
                    return TurnReservation(identifier, [], cached_response=response)
                if (existing["status"] == "running" and existing["lease_expires_at"] is not None and
                        existing["lease_expires_at"] > current_time):
                    raise ConversationConflict()
            connection.execute("""UPDATE turns SET status='failed',lease_token=NULL,lease_expires_at=NULL
                WHERE workspace_id=? AND principal_id=? AND conversation_id=? AND status='running'
                  AND (lease_expires_at IS NULL OR lease_expires_at<=?)""",
                (workspace, principal, identifier, current_time))
            running = connection.execute("""SELECT 1 FROM turns
                WHERE workspace_id=? AND principal_id=? AND conversation_id=? AND status='running'""",
                                         (workspace, principal, identifier)).fetchone()
            if running is not None:
                raise ConversationConflict()
            history = self._history(connection, workspace, principal, identifier)
            if existing is None:
                sequence = connection.execute("""SELECT COALESCE(MAX(sequence),0)+1 FROM turns
                    WHERE workspace_id=? AND principal_id=? AND conversation_id=?""",
                                              (workspace, principal, identifier)).fetchone()[0]
                connection.execute("""INSERT INTO turns(workspace_id,principal_id,conversation_id,sequence,
                    idempotency_key,question,status,lease_token,lease_expires_at)
                    VALUES (?,?,?,?,?,?,'running',?,?)""",
                    (workspace, principal, identifier, sequence, idempotency_key, question,
                     lease_token, lease_expires_at))
            else:
                connection.execute("""UPDATE turns SET status='running',lease_token=?,lease_expires_at=?
                    WHERE workspace_id=? AND principal_id=?
                    AND conversation_id=? AND idempotency_key=?""",
                    (lease_token, lease_expires_at, workspace, principal, identifier, idempotency_key))
            connection.execute("""UPDATE conversations SET updated_at=CURRENT_TIMESTAMP
                WHERE workspace_id=? AND principal_id=? AND conversation_id=?""",
                (workspace, principal, identifier))
            connection.commit()
        return TurnReservation(identifier, history, lease_token=lease_token)

    def complete(self, workspace: str, principal: str, conversation_id: str,
                 idempotency_key: str, lease_token: str, response: dict,
                 *, now: int | None = None) -> None:
        encoded = json.dumps(response, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        current_time = int(time.time() if now is None else now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute("""UPDATE turns SET status='complete',response_json=?,corpus_version=?,
                lease_token=NULL,lease_expires_at=NULL
                WHERE workspace_id=? AND principal_id=? AND conversation_id=? AND idempotency_key=?
                  AND status='running' AND lease_token=? AND lease_expires_at>?""",
                (encoded, response.get("corpus_version"), workspace, principal, conversation_id,
                 idempotency_key, lease_token, current_time))
            if updated.rowcount != 1:
                raise ConversationConflict()
            connection.execute("""UPDATE conversations SET updated_at=CURRENT_TIMESTAMP
                WHERE workspace_id=? AND principal_id=? AND conversation_id=?""",
                (workspace, principal, conversation_id))
            self._refresh_summary(connection, workspace, principal, conversation_id)
            connection.commit()

    def renew(self, workspace: str, principal: str, conversation_id: str,
              idempotency_key: str, lease_token: str, *, now: int | None = None) -> None:
        current_time = int(time.time() if now is None else now)
        with self._connect() as connection:
            updated = connection.execute("""UPDATE turns SET lease_expires_at=?
                WHERE workspace_id=? AND principal_id=? AND conversation_id=? AND idempotency_key=?
                  AND status='running' AND lease_token=? AND lease_expires_at>?""",
                (current_time + self.lease_seconds, workspace, principal, conversation_id,
                 idempotency_key, lease_token, current_time))
            connection.commit()
            if updated.rowcount != 1:
                raise ConversationConflict()

    def fail(self, workspace: str, principal: str, conversation_id: str,
             idempotency_key: str, lease_token: str) -> None:
        with self._connect() as connection:
            connection.execute("""UPDATE turns SET status='failed',lease_token=NULL,lease_expires_at=NULL
                WHERE workspace_id=? AND principal_id=? AND conversation_id=? AND idempotency_key=?
                  AND status='running' AND lease_token=?""",
                (workspace, principal, conversation_id, idempotency_key, lease_token))
            connection.commit()

    def recover_expired(self, *, now: int | None = None) -> int:
        """Release only expired leases; live turns owned by other workers remain valid."""
        current_time = int(time.time() if now is None else now)
        with self._connect() as connection:
            updated = connection.execute("""UPDATE turns
                SET status='failed',lease_token=NULL,lease_expires_at=NULL
                WHERE status='running' AND (lease_expires_at IS NULL OR lease_expires_at<=?)""",
                (current_time,))
            connection.commit()
            return updated.rowcount

    def recover_interrupted(self) -> int:
        """Backward-compatible alias with multi-worker-safe expiry semantics."""
        return self.recover_expired()

    def delete(self, workspace: str, principal: str, conversation_id: str) -> None:
        self._validate(workspace, principal, conversation_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute("""SELECT 1 FROM conversations
                WHERE workspace_id=? AND principal_id=? AND conversation_id=?""",
                (workspace, principal, conversation_id)).fetchone()
            if exists is None:
                raise ConversationNotFound()
            values = (workspace, principal, conversation_id)
            connection.execute("""DELETE FROM conversation_summaries
                WHERE workspace_id=? AND principal_id=? AND conversation_id=?""", values)
            connection.execute("""DELETE FROM turns
                WHERE workspace_id=? AND principal_id=? AND conversation_id=?""", values)
            connection.execute("""DELETE FROM conversations
                WHERE workspace_id=? AND principal_id=? AND conversation_id=?""", values)
            connection.commit()

    def purge_expired(self, *, now: float | None = None) -> int:
        current_time = time.time() if now is None else now
        cutoff = datetime.fromtimestamp(
            current_time - self.retention_days * 24 * 60 * 60, timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute("""SELECT workspace_id,principal_id,conversation_id
                FROM conversations c WHERE c.updated_at<? AND NOT EXISTS (
                    SELECT 1 FROM turns t WHERE t.workspace_id=c.workspace_id
                    AND t.principal_id=c.principal_id AND t.conversation_id=c.conversation_id
                    AND t.status='running')""", (cutoff,)).fetchall()
            for row in rows:
                values = (row["workspace_id"], row["principal_id"], row["conversation_id"])
                connection.execute("""DELETE FROM conversation_summaries
                    WHERE workspace_id=? AND principal_id=? AND conversation_id=?""", values)
                connection.execute("""DELETE FROM turns
                    WHERE workspace_id=? AND principal_id=? AND conversation_id=?""", values)
                connection.execute("""DELETE FROM conversations
                    WHERE workspace_id=? AND principal_id=? AND conversation_id=?""", values)
            connection.commit()
            return len(rows)

    def get(self, workspace: str, principal: str, conversation_id: str, *,
            limit: int = DETAIL_TURN_LIMIT, before_sequence: int | None = None) -> dict:
        self._validate(workspace, principal, conversation_id)
        if not 1 <= limit <= 100:
            raise ValueError("Conversation page limit must be between 1 and 100")
        if before_sequence is not None and before_sequence <= 0:
            raise ValueError("Conversation cursor must be positive")
        with self._connect() as connection:
            row = connection.execute("""SELECT title,created_at,updated_at FROM conversations
                WHERE workspace_id=? AND principal_id=? AND conversation_id=?""",
                (workspace, principal, conversation_id)).fetchone()
            if row is None:
                raise ConversationNotFound()
            turns = connection.execute("""SELECT sequence,question,response_json FROM turns
                WHERE workspace_id=? AND principal_id=? AND conversation_id=? AND status='complete'
                  AND (? IS NULL OR sequence<?)
                ORDER BY sequence DESC LIMIT ?""",
                (workspace, principal, conversation_id, before_sequence, before_sequence, limit + 1)).fetchall()
            summary = connection.execute("""SELECT through_sequence FROM conversation_summaries
                WHERE workspace_id=? AND principal_id=? AND conversation_id=?""",
                (workspace, principal, conversation_id)).fetchone()
        has_more = len(turns) > limit
        turns = turns[:limit]
        messages = []
        for turn in reversed(turns):
            response = json.loads(turn["response_json"])
            messages.extend([{"role": "user", "content": turn["question"], "sequence": turn["sequence"]},
                             {"role": "assistant", "content": response["response"], "response": response}])
        return {"conversation_id": conversation_id, "title": row["title"],
                "created_at": row["created_at"], "updated_at": row["updated_at"], "messages": messages,
                "next_before_sequence": turns[-1]["sequence"] if has_more and turns else None,
                "summarized_through_sequence": summary["through_sequence"] if summary else None}

    def list_recent(self, workspace: str, principal: str, limit: int = 20) -> list[dict]:
        self._validate(workspace, principal)
        if not 1 <= limit <= 100:
            raise ValueError("Conversation list limit must be between 1 and 100")
        with self._connect() as connection:
            rows = connection.execute("""SELECT c.conversation_id,c.title,c.updated_at FROM conversations c
                WHERE c.workspace_id=? AND c.principal_id=? AND EXISTS (
                    SELECT 1 FROM turns t WHERE t.workspace_id=c.workspace_id AND t.principal_id=c.principal_id
                    AND t.conversation_id=c.conversation_id AND t.status='complete')
                ORDER BY c.updated_at DESC,c.rowid DESC LIMIT ?""", (workspace, principal, limit)).fetchall()
        return [dict(row) for row in rows]
