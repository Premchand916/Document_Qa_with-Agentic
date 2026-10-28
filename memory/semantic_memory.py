"""Explicit, owner-scoped semantic facts and preferences; never auto-extracted."""

from contextlib import contextmanager
from pathlib import Path
import re
import sqlite3
import time
from uuid import uuid4

from storage.workspaces import workspace_key


SEMANTIC_CATEGORIES = frozenset({"preference", "profile", "constraint", "terminology"})
_QUERY_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "in",
    "is", "it", "me", "my", "of", "on", "or", "please", "the", "this", "to", "what",
    "when", "where", "which", "who", "why", "with", "you",
})
_FORBIDDEN_KEY = re.compile(
    r"\b(password|passcode|secret|api[ _-]?key|access[ _-]?token|private[ _-]?key|credential)\b",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:sk|ghp|xox[baprs])[-_][A-Za-z0-9_-]{16,}\b|"
    r"\bAKIA[A-Z0-9]{16}\b",
    re.IGNORECASE,
)
_INSTRUCTION_OVERRIDE = re.compile(
    r"\b(ignore|disregard|override|bypass)\b.{0,40}\b(system|developer|previous|security|policy|instructions?)\b|"
    r"\b(reveal|print|expose)\b.{0,30}\b(system prompt|hidden instructions?|credentials?|secrets?)\b",
    re.IGNORECASE,
)


class SemanticMemoryNotFound(Exception):
    """Memory is missing or outside authorized workspace/principal."""


class SemanticMemoryRejected(ValueError):
    """Memory contains prohibited secret-like or instruction-override content."""


def _normalized_key(value: str) -> tuple[str, str]:
    display = " ".join(value.strip().split())
    if not 1 <= len(display) <= 64 or not re.fullmatch(r"[\w][\w .:/-]*", display, re.UNICODE):
        raise ValueError("Semantic memory key must contain 1 to 64 safe characters")
    return display, display.casefold()


def _validated_value(key: str, value: str) -> str:
    normalized = " ".join(value.strip().split())
    if not 1 <= len(normalized) <= 1000:
        raise ValueError("Semantic memory value must contain 1 to 1000 characters")
    if _FORBIDDEN_KEY.search(key) or _SECRET_VALUE.search(normalized):
        raise SemanticMemoryRejected("Secrets and credentials cannot be stored as semantic memory")
    if _INSTRUCTION_OVERRIDE.search(normalized):
        raise SemanticMemoryRejected("Instruction overrides cannot be stored as semantic memory")
    return normalized


def _query_terms(value: str | None) -> set[str]:
    if not value:
        return set()
    return {
        token for token in re.findall(r"[^\W_]+", value.casefold(), re.UNICODE)
        if len(token) > 1 and token not in _QUERY_STOPWORDS
    }


def _relevance_score(row: sqlite3.Row, query_terms: set[str]) -> int:
    if not query_terms:
        return 0
    key_terms = _query_terms(row["display_key"])
    value_terms = _query_terms(row["memory_value"])
    return 4 * len(query_terms & key_terms) + len(query_terms & value_terms)


class SemanticMemoryRepository:
    MAX_ACTIVE = 50
    RECALL_LIMIT = 20
    RECALL_CHARACTERS = 6000

    def __init__(self, root: Path, *, retention_days: int = 365):
        self.root = Path(root)
        if not 1 <= retention_days <= 3650:
            raise ValueError("Semantic memory retention must be between 1 and 3650 days")
        self.retention_days = retention_days

    @contextmanager
    def _connect(self):
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        path = self.root / "conversations.sqlite3"
        path.touch(mode=0o600, exist_ok=True)
        connection = sqlite3.connect(path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=10000")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS semantic_memories (
                    workspace_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL, category TEXT NOT NULL
                      CHECK(category IN ('preference','profile','constraint','terminology')),
                    memory_key TEXT NOT NULL, display_key TEXT NOT NULL, memory_value TEXT NOT NULL,
                    source TEXT NOT NULL CHECK(source='user_explicit'),
                    created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, expires_at INTEGER NOT NULL,
                    PRIMARY KEY(workspace_id,principal_id,memory_id),
                    UNIQUE(workspace_id,principal_id,category,memory_key));
                CREATE INDEX IF NOT EXISTS semantic_memory_recall
                    ON semantic_memories(workspace_id,principal_id,expires_at,updated_at);
                CREATE INDEX IF NOT EXISTS semantic_memory_expiry ON semantic_memories(expires_at);
            """)
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _validate_owner(workspace: str, principal: str, memory_id: str | None = None) -> None:
        workspace_key(workspace)
        if not principal or len(principal) > 256:
            raise ValueError("Invalid principal")
        if memory_id is not None and not re.fullmatch(r"[a-f0-9]{32}", memory_id):
            raise ValueError("Invalid semantic memory identifier")

    def upsert(self, workspace: str, principal: str, category: str, key: str, value: str,
               *, now: int | None = None) -> dict:
        self._validate_owner(workspace, principal)
        if category not in SEMANTIC_CATEGORIES:
            raise ValueError("Invalid semantic memory category")
        display_key, memory_key = _normalized_key(key)
        memory_value = _validated_value(display_key, value)
        current_time = int(time.time() if now is None else now)
        expires_at = current_time + self.retention_days * 24 * 60 * 60
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("""SELECT memory_id,created_at FROM semantic_memories
                WHERE workspace_id=? AND principal_id=? AND category=? AND memory_key=?""",
                (workspace, principal, category, memory_key)).fetchone()
            if existing is None:
                active = connection.execute("""SELECT COUNT(*) FROM semantic_memories
                    WHERE workspace_id=? AND principal_id=? AND expires_at>?""",
                    (workspace, principal, current_time)).fetchone()[0]
                if active >= self.MAX_ACTIVE:
                    raise ValueError(f"Semantic memory limit is {self.MAX_ACTIVE}")
                memory_id, created_at = uuid4().hex, current_time
                connection.execute("""INSERT INTO semantic_memories(
                    workspace_id,principal_id,memory_id,category,memory_key,display_key,memory_value,
                    source,created_at,updated_at,expires_at)
                    VALUES (?,?,?,?,?,?,?,'user_explicit',?,?,?)""",
                    (workspace, principal, memory_id, category, memory_key, display_key, memory_value,
                     created_at, current_time, expires_at))
            else:
                memory_id, created_at = existing["memory_id"], existing["created_at"]
                connection.execute("""UPDATE semantic_memories
                    SET display_key=?,memory_value=?,updated_at=?,expires_at=?
                    WHERE workspace_id=? AND principal_id=? AND memory_id=?""",
                    (display_key, memory_value, current_time, expires_at,
                     workspace, principal, memory_id))
            connection.commit()
        return {"memory_id": memory_id, "category": category, "key": display_key,
                "value": memory_value, "source": "user_explicit", "created_at": created_at,
                "updated_at": current_time, "expires_at": expires_at}

    def recall(self, workspace: str, principal: str, query: str | None = None,
               *, now: int | None = None) -> list[dict]:
        self._validate_owner(workspace, principal)
        current_time = int(time.time() if now is None else now)
        with self._connect() as connection:
            rows = connection.execute("""SELECT memory_id,category,display_key,memory_value,source,
                created_at,updated_at,expires_at FROM semantic_memories
                WHERE workspace_id=? AND principal_id=? AND expires_at>?
                ORDER BY updated_at DESC,memory_id DESC LIMIT ?""",
                (workspace, principal, current_time, self.MAX_ACTIVE)).fetchall()
        query_terms = _query_terms(query)
        ranked = sorted(
            ((row, _relevance_score(row, query_terms)) for row in rows),
            key=lambda item: (item[1], item[0]["updated_at"], item[0]["memory_id"]),
            reverse=True,
        )[:self.RECALL_LIMIT]
        result, characters = [], 0
        for row, score in ranked:
            size = len(row["display_key"]) + len(row["memory_value"])
            if characters + size > self.RECALL_CHARACTERS:
                continue
            characters += size
            result.append({"memory_id": row["memory_id"], "category": row["category"],
                           "key": row["display_key"], "value": row["memory_value"],
                           "source": row["source"], "created_at": row["created_at"],
                           "updated_at": row["updated_at"], "expires_at": row["expires_at"],
                           "relevance_score": score})
        return result

    def list_recent(self, workspace: str, principal: str, *, now: int | None = None,
                    limit: int = 50) -> list[dict]:
        self._validate_owner(workspace, principal)
        if not 1 <= limit <= 100:
            raise ValueError("Semantic memory list limit must be between 1 and 100")
        current_time = int(time.time() if now is None else now)
        with self._connect() as connection:
            rows = connection.execute("""SELECT memory_id,category,display_key,memory_value,source,
                created_at,updated_at,expires_at FROM semantic_memories
                WHERE workspace_id=? AND principal_id=? AND expires_at>?
                ORDER BY updated_at DESC,memory_id DESC LIMIT ?""",
                (workspace, principal, current_time, limit)).fetchall()
        return [{"memory_id": row["memory_id"], "category": row["category"],
                 "key": row["display_key"], "value": row["memory_value"],
                 "source": row["source"], "created_at": row["created_at"],
                 "updated_at": row["updated_at"], "expires_at": row["expires_at"]}
                for row in rows]

    def export(self, workspace: str, principal: str, *, now: int | None = None) -> dict:
        current_time = int(time.time() if now is None else now)
        return {
            "schema_version": 1,
            "kind": "semantic_memory",
            "exported_at": current_time,
            "workspace_id": workspace,
            "memories": self.list_recent(
                workspace, principal, now=current_time, limit=self.MAX_ACTIVE,
            ),
        }

    def delete(self, workspace: str, principal: str, memory_id: str) -> None:
        self._validate_owner(workspace, principal, memory_id)
        with self._connect() as connection:
            deleted = connection.execute("""DELETE FROM semantic_memories
                WHERE workspace_id=? AND principal_id=? AND memory_id=?""",
                (workspace, principal, memory_id))
            connection.commit()
            if deleted.rowcount != 1:
                raise SemanticMemoryNotFound()

    def purge_expired(self, *, now: int | None = None) -> int:
        current_time = int(time.time() if now is None else now)
        with self._connect() as connection:
            deleted = connection.execute("DELETE FROM semantic_memories WHERE expires_at<=?", (current_time,))
            connection.commit()
            return deleted.rowcount
