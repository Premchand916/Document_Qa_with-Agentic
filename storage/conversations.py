"""Workspace- and principal-owned durable chat turns.

Only serializable messages and response values enter this store. Graph objects,
FAISS indexes, DataFrames, and provider clients remain request-scoped.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
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
    cached_response: dict | None = None


class ConversationRepository:
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
                    PRIMARY KEY(workspace_id, principal_id, conversation_id, sequence),
                    UNIQUE(workspace_id, principal_id, conversation_id, idempotency_key),
                    FOREIGN KEY(workspace_id, principal_id, conversation_id)
                      REFERENCES conversations(workspace_id, principal_id, conversation_id));
                CREATE UNIQUE INDEX IF NOT EXISTS turns_principal_request
                    ON turns(workspace_id, principal_id, idempotency_key);
            """)
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

    @staticmethod
    def _history(connection, workspace: str, principal: str, conversation_id: str) -> list[dict]:
        rows = connection.execute("""SELECT question, response_json FROM turns
            WHERE workspace_id=? AND principal_id=? AND conversation_id=? AND status='complete'
            ORDER BY sequence DESC LIMIT 8""", (workspace, principal, conversation_id)).fetchall()
        messages = []
        for row in reversed(rows):
            answer = json.loads(row["response_json"])["response"]
            messages.extend([{"role": "user", "content": row["question"][:4000]},
                             {"role": "assistant", "content": str(answer)[:4000]}])
        while sum(len(message["content"]) for message in messages) > 16000:
            messages = messages[2:]
        return messages

    def reserve(self, workspace: str, principal: str, conversation_id: str | None,
                idempotency_key: str, question: str) -> TurnReservation:
        self._validate(workspace, principal, conversation_id)
        if not idempotency_key or len(idempotency_key) > 128:
            raise ValueError("Invalid idempotency key")
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
            existing = connection.execute("""SELECT question,status,response_json FROM turns
                WHERE workspace_id=? AND principal_id=? AND conversation_id=? AND idempotency_key=?""",
                                          (workspace, principal, identifier, idempotency_key)).fetchone()
            if existing is not None:
                if existing["question"] != question:
                    raise ConversationConflict()
                if existing["status"] == "complete":
                    response = json.loads(existing["response_json"])
                    connection.commit()
                    return TurnReservation(identifier, [], response)
                if existing["status"] == "running":
                    raise ConversationConflict()
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
                    idempotency_key,question,status) VALUES (?,?,?,?,?,?,'running')""",
                    (workspace, principal, identifier, sequence, idempotency_key, question))
            else:
                connection.execute("""UPDATE turns SET status='running' WHERE workspace_id=? AND principal_id=?
                    AND conversation_id=? AND idempotency_key=?""",
                    (workspace, principal, identifier, idempotency_key))
            connection.commit()
        return TurnReservation(identifier, history)

    def complete(self, workspace: str, principal: str, conversation_id: str,
                 idempotency_key: str, response: dict) -> None:
        encoded = json.dumps(response, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute("""UPDATE turns SET status='complete',response_json=?,corpus_version=?
                WHERE workspace_id=? AND principal_id=? AND conversation_id=? AND idempotency_key=? AND status='running'""",
                (encoded, response.get("corpus_version"), workspace, principal, conversation_id, idempotency_key))
            if updated.rowcount != 1:
                raise ConversationConflict()
            connection.execute("""UPDATE conversations SET updated_at=CURRENT_TIMESTAMP
                WHERE workspace_id=? AND principal_id=? AND conversation_id=?""",
                (workspace, principal, conversation_id))
            connection.commit()

    def fail(self, workspace: str, principal: str, conversation_id: str, idempotency_key: str) -> None:
        with self._connect() as connection:
            connection.execute("""UPDATE turns SET status='failed' WHERE workspace_id=? AND principal_id=?
                AND conversation_id=? AND idempotency_key=? AND status='running'""",
                (workspace, principal, conversation_id, idempotency_key))
            connection.commit()

    def recover_interrupted(self) -> int:
        """Release turns left running by a previous single-worker API process."""
        with self._connect() as connection:
            updated = connection.execute("UPDATE turns SET status='failed' WHERE status='running'")
            connection.commit()
            return updated.rowcount

    def get(self, workspace: str, principal: str, conversation_id: str) -> dict:
        self._validate(workspace, principal, conversation_id)
        with self._connect() as connection:
            row = connection.execute("""SELECT title,created_at,updated_at FROM conversations
                WHERE workspace_id=? AND principal_id=? AND conversation_id=?""",
                (workspace, principal, conversation_id)).fetchone()
            if row is None:
                raise ConversationNotFound()
            turns = connection.execute("""SELECT question,response_json FROM turns
                WHERE workspace_id=? AND principal_id=? AND conversation_id=? AND status='complete'
                ORDER BY sequence""", (workspace, principal, conversation_id)).fetchall()
        messages = []
        for turn in turns:
            response = json.loads(turn["response_json"])
            messages.extend([{"role": "user", "content": turn["question"]},
                             {"role": "assistant", "content": response["response"], "response": response}])
        return {"conversation_id": conversation_id, "title": row["title"],
                "created_at": row["created_at"], "updated_at": row["updated_at"], "messages": messages}

    def list_recent(self, workspace: str, principal: str, limit: int = 20) -> list[dict]:
        self._validate(workspace, principal)
        with self._connect() as connection:
            rows = connection.execute("""SELECT c.conversation_id,c.title,c.updated_at FROM conversations c
                WHERE c.workspace_id=? AND c.principal_id=? AND EXISTS (
                    SELECT 1 FROM turns t WHERE t.workspace_id=c.workspace_id AND t.principal_id=c.principal_id
                    AND t.conversation_id=c.conversation_id AND t.status='complete')
                ORDER BY c.updated_at DESC,c.rowid DESC LIMIT ?""", (workspace, principal, limit)).fetchall()
        return [dict(row) for row in rows]
