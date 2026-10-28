"""Opt-in, owner-scoped feedback episodes derived from completed chat turns."""

from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3
import time
from uuid import uuid4

from storage.conversations import ConversationNotFound
from storage.workspaces import workspace_key


class EpisodeNotFound(Exception):
    """Episode is missing or outside the authorized workspace/principal."""


class EpisodeConflict(Exception):
    """Feedback already exists for the selected completed turn."""


def normalize_task_key(question: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", question.casefold())
    return " ".join(tokens)[:512]


class EpisodicMemoryRepository:
    def __init__(self, root: Path, *, retention_days: int = 180):
        self.root = Path(root)
        if not 1 <= retention_days <= 3650:
            raise ValueError("Episode retention must be between 1 and 3650 days")
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
                CREATE TABLE IF NOT EXISTS episodes (
                    workspace_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL, task_key TEXT NOT NULL,
                    correction TEXT NOT NULL,
                    validation_status TEXT NOT NULL
                      CHECK(validation_status IN ('unverified','verified','rejected')),
                    corpus_version TEXT,
                    source_ids_json TEXT NOT NULL, tool_context_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL, result_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL,
                    PRIMARY KEY(workspace_id, principal_id, episode_id),
                    UNIQUE(workspace_id, principal_id, conversation_id, idempotency_key),
                    FOREIGN KEY(workspace_id, principal_id, conversation_id)
                      REFERENCES conversations(workspace_id, principal_id, conversation_id)
                      ON DELETE CASCADE);
                CREATE INDEX IF NOT EXISTS episodes_recall
                    ON episodes(workspace_id,principal_id,task_key,validation_status,corpus_version,expires_at);
                CREATE INDEX IF NOT EXISTS episodes_expiry ON episodes(expires_at);
                CREATE TABLE IF NOT EXISTS episode_reviews (
                    workspace_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL, review_id TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL, previous_status TEXT NOT NULL,
                    decision TEXT NOT NULL CHECK(decision IN ('verified','rejected')),
                    review_note TEXT NOT NULL, reviewed_at INTEGER NOT NULL,
                    PRIMARY KEY(workspace_id,principal_id,episode_id,review_id),
                    FOREIGN KEY(workspace_id,principal_id,episode_id)
                      REFERENCES episodes(workspace_id,principal_id,episode_id)
                      ON DELETE CASCADE);
                CREATE INDEX IF NOT EXISTS episode_review_history
                    ON episode_reviews(workspace_id,principal_id,episode_id,reviewed_at);
            """)
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _validate(workspace: str, principal: str, identifier: str | None = None) -> None:
        workspace_key(workspace)
        if not principal or len(principal) > 256:
            raise ValueError("Invalid principal")
        if identifier is not None and not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("Invalid identifier")

    def save_feedback(self, workspace: str, principal: str, conversation_id: str,
                      idempotency_key: str, correction: str, *, now: int | None = None) -> dict:
        self._validate(workspace, principal, conversation_id)
        correction = correction.strip()
        if not 1 <= len(correction) <= 4000:
            raise ValueError("Correction must contain 1 to 4000 characters")
        current_time = int(time.time() if now is None else now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("""SELECT question,response_json FROM turns
                WHERE workspace_id=? AND principal_id=? AND conversation_id=?
                  AND idempotency_key=? AND status='complete'""",
                (workspace, principal, conversation_id, idempotency_key)).fetchone()
            if row is None:
                conversation = connection.execute("""SELECT 1 FROM conversations
                    WHERE workspace_id=? AND principal_id=? AND conversation_id=?""",
                    (workspace, principal, conversation_id)).fetchone()
                if conversation is None:
                    raise ConversationNotFound()
                raise EpisodeNotFound("Completed turn not found")
            response = json.loads(row["response_json"])
            documents = response.get("documents") or []
            source_ids = sorted({
                str((document.get("metadata") or {}).get("source_id") or
                    (document.get("metadata") or {}).get("source") or "").strip()
                for document in documents
                if isinstance(document, dict)
            } - {""})
            evidence = {"corpus_version": response.get("corpus_version"),
                        "citations": response.get("citations") or []}
            tool_context = {"response_plan": response.get("response_plan") or {},
                            "tool_result": response.get("tool_result")}
            result = {"response": response.get("response", ""),
                      "result_status": response.get("result_status"),
                      "tool_result": response.get("tool_result")}
            episode_id = uuid4().hex
            expires_at = current_time + self.retention_days * 24 * 60 * 60
            try:
                connection.execute("""INSERT INTO episodes(
                    workspace_id,principal_id,episode_id,conversation_id,idempotency_key,task_key,
                    correction,validation_status,corpus_version,source_ids_json,tool_context_json,
                    evidence_json,result_json,created_at,expires_at)
                    VALUES (?,?,?,?,?,?,?,'unverified',?,?,?,?,?,?,?)""",
                    (workspace, principal, episode_id, conversation_id, idempotency_key,
                     normalize_task_key(row["question"]), correction, response.get("corpus_version"),
                     json.dumps(source_ids, separators=(",", ":")),
                     json.dumps(tool_context, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                     json.dumps(evidence, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                     json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                     current_time, expires_at))
            except sqlite3.IntegrityError as exc:
                raise EpisodeConflict() from exc
            connection.commit()
        return {"episode_id": episode_id, "conversation_id": conversation_id,
                "validation_status": "unverified", "expires_at": expires_at}

    def review(self, workspace: str, principal: str, episode_id: str, status: str,
               reviewer_id: str, *, note: str = "", now: int | None = None) -> dict:
        """Operator validation boundary; intentionally not exposed as a public API route."""
        self._validate(workspace, principal, episode_id)
        if status not in {"verified", "rejected"}:
            raise ValueError("Episode status must be verified or rejected")
        reviewer_id = reviewer_id.strip()
        note = note.strip()
        if not 1 <= len(reviewer_id) <= 256:
            raise ValueError("Reviewer identifier must contain 1 to 256 characters")
        if len(note) > 2000:
            raise ValueError("Review note must contain at most 2000 characters")
        reviewed_at = int(time.time() if now is None else now)
        review_id = uuid4().hex
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("""SELECT validation_status FROM episodes
                WHERE workspace_id=? AND principal_id=? AND episode_id=?""",
                (workspace, principal, episode_id)).fetchone()
            if current is None:
                raise EpisodeNotFound()
            previous_status = current["validation_status"]
            connection.execute("""UPDATE episodes SET validation_status=?
                WHERE workspace_id=? AND principal_id=? AND episode_id=?""",
                (status, workspace, principal, episode_id))
            connection.execute("""INSERT INTO episode_reviews(
                workspace_id,principal_id,episode_id,review_id,reviewer_id,
                previous_status,decision,review_note,reviewed_at)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (workspace, principal, episode_id, review_id, reviewer_id,
                 previous_status, status, note, reviewed_at))
            connection.commit()
        return {"review_id": review_id, "episode_id": episode_id,
                "reviewer_id": reviewer_id, "previous_status": previous_status,
                "decision": status, "review_note": note, "reviewed_at": reviewed_at}

    def set_status(self, workspace: str, principal: str, episode_id: str, status: str) -> None:
        """Backward-compatible internal validation helper with an auditable operator identity."""
        self.review(workspace, principal, episode_id, status, "internal-operator")

    def list_reviews(self, workspace: str, principal: str, episode_id: str) -> list[dict]:
        self._validate(workspace, principal, episode_id)
        with self._connect() as connection:
            exists = connection.execute("""SELECT 1 FROM episodes
                WHERE workspace_id=? AND principal_id=? AND episode_id=?""",
                (workspace, principal, episode_id)).fetchone()
            if exists is None:
                raise EpisodeNotFound()
            rows = connection.execute("""SELECT review_id,reviewer_id,previous_status,
                decision,review_note,reviewed_at FROM episode_reviews
                WHERE workspace_id=? AND principal_id=? AND episode_id=?
                ORDER BY reviewed_at ASC,review_id ASC""",
                (workspace, principal, episode_id)).fetchall()
        return [dict(row) for row in rows]

    def recall(self, workspace: str, principal: str, question: str,
               corpus_version: str | None, *, now: int | None = None, limit: int = 3) -> list[dict]:
        self._validate(workspace, principal)
        if not 1 <= limit <= 5:
            raise ValueError("Episode recall limit must be between 1 and 5")
        current_time = int(time.time() if now is None else now)
        with self._connect() as connection:
            rows = connection.execute("""SELECT episode_id,correction,source_ids_json,
                tool_context_json,evidence_json,result_json,created_at,expires_at
                FROM episodes WHERE workspace_id=? AND principal_id=? AND task_key=?
                  AND validation_status='verified' AND expires_at>?
                  AND ((corpus_version IS NULL AND ? IS NULL) OR corpus_version=?)
                ORDER BY created_at DESC,episode_id DESC LIMIT ?""",
                (workspace, principal, normalize_task_key(question), current_time,
                 corpus_version, corpus_version, limit)).fetchall()
        return [{"episode_id": row["episode_id"], "correction": row["correction"],
                 "source_ids": json.loads(row["source_ids_json"]),
                 "tool_context": json.loads(row["tool_context_json"]),
                 "evidence": json.loads(row["evidence_json"]),
                 "result": json.loads(row["result_json"]),
                 "created_at": row["created_at"], "expires_at": row["expires_at"]}
                for row in rows]

    def list_recent(self, workspace: str, principal: str, *, limit: int = 50) -> list[dict]:
        self._validate(workspace, principal)
        if not 1 <= limit <= 100:
            raise ValueError("Episode list limit must be between 1 and 100")
        with self._connect() as connection:
            rows = connection.execute("""SELECT episode_id,conversation_id,idempotency_key,task_key,
                correction,validation_status,created_at,expires_at FROM episodes
                WHERE workspace_id=? AND principal_id=? ORDER BY created_at DESC,episode_id DESC LIMIT ?""",
                (workspace, principal, limit)).fetchall()
        return [dict(row) for row in rows]

    def delete(self, workspace: str, principal: str, episode_id: str) -> None:
        self._validate(workspace, principal, episode_id)
        with self._connect() as connection:
            deleted = connection.execute("""DELETE FROM episodes
                WHERE workspace_id=? AND principal_id=? AND episode_id=?""",
                (workspace, principal, episode_id))
            connection.commit()
            if deleted.rowcount != 1:
                raise EpisodeNotFound()

    def purge_expired(self, *, now: int | None = None) -> int:
        current_time = int(time.time() if now is None else now)
        with self._connect() as connection:
            deleted = connection.execute("DELETE FROM episodes WHERE expires_at<=?", (current_time,))
            connection.commit()
            return deleted.rowcount
