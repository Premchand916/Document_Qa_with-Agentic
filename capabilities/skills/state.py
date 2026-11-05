"""Workspace-scoped skill enablement with auditable state changes."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

from domain.skills import SkillDescriptor
from storage.workspaces import workspace_key


class SkillStateRepository:
    def __init__(self, root: Path):
        self.root = Path(root)

    @contextmanager
    def _connect(self):
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        path = self.root / "skills.sqlite3"
        path.touch(mode=0o600, exist_ok=True)
        connection = sqlite3.connect(path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=10000")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS skill_enablement (
                    workspace_id TEXT NOT NULL, skill_id TEXT NOT NULL,
                    version TEXT NOT NULL, digest TEXT NOT NULL, enabled INTEGER NOT NULL
                      CHECK(enabled IN (0,1)), updated_by TEXT NOT NULL, updated_at INTEGER NOT NULL,
                    PRIMARY KEY(workspace_id,skill_id));
                CREATE TABLE IF NOT EXISTS skill_state_audit (
                    workspace_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                    event_id TEXT NOT NULL, skill_id TEXT NOT NULL, version TEXT NOT NULL,
                    digest TEXT NOT NULL, action TEXT NOT NULL CHECK(action IN ('enabled','disabled')),
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY(workspace_id,event_id));
                CREATE INDEX IF NOT EXISTS skill_state_audit_recent
                    ON skill_state_audit(workspace_id,created_at,event_id);
            """)
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _validate_owner(workspace: str, principal: str) -> None:
        workspace_key(workspace)
        if not principal or len(principal) > 256:
            raise ValueError("Invalid principal")

    def status(self, workspace: str, descriptor: SkillDescriptor) -> tuple[bool, str]:
        workspace_key(workspace)
        with self._connect() as connection:
            row = connection.execute("""SELECT version,digest,enabled FROM skill_enablement
                WHERE workspace_id=? AND skill_id=?""",
                (workspace, descriptor.manifest.skill_id)).fetchone()
        if row is None:
            return descriptor.manifest.enabled_by_default, "package_default"
        if row["version"] != descriptor.manifest.version or row["digest"] != descriptor.digest:
            return False, "changed_since_approval"
        return bool(row["enabled"]), "workspace_setting"

    def set_enabled(self, workspace: str, principal: str, descriptor: SkillDescriptor,
                    enabled: bool, *, now: int | None = None) -> dict:
        self._validate_owner(workspace, principal)
        if not isinstance(enabled, bool):
            raise ValueError("Skill enabled state must be boolean")
        current_time = int(time.time() if now is None else now)
        action = "enabled" if enabled else "disabled"
        event_id = uuid4().hex
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("""SELECT version,digest,enabled FROM skill_enablement
                WHERE workspace_id=? AND skill_id=?""",
                (workspace, descriptor.manifest.skill_id)).fetchone()
            if (existing is None or existing["version"] != descriptor.manifest.version
                    or existing["digest"] != descriptor.digest
                    or bool(existing["enabled"]) != enabled):
                connection.execute("""INSERT INTO skill_enablement(
                    workspace_id,skill_id,version,digest,enabled,updated_by,updated_at)
                    VALUES (?,?,?,?,?,?,?) ON CONFLICT(workspace_id,skill_id) DO UPDATE SET
                    version=excluded.version,digest=excluded.digest,enabled=excluded.enabled,
                    updated_by=excluded.updated_by,updated_at=excluded.updated_at""",
                    (workspace, descriptor.manifest.skill_id, descriptor.manifest.version,
                     descriptor.digest, int(enabled), principal, current_time))
                connection.execute("""INSERT INTO skill_state_audit(
                    workspace_id,principal_id,event_id,skill_id,version,digest,action,created_at)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (workspace, principal, event_id, descriptor.manifest.skill_id,
                     descriptor.manifest.version, descriptor.digest, action, current_time))
            connection.commit()
        return {"skill_id": descriptor.manifest.skill_id, "enabled": enabled,
                "version": descriptor.manifest.version, "digest": descriptor.digest}

    def audit(self, workspace: str, *, limit: int = 50) -> list[dict]:
        workspace_key(workspace)
        if not 1 <= limit <= 100:
            raise ValueError("Skill audit limit must be between 1 and 100")
        with self._connect() as connection:
            rows = connection.execute("""SELECT principal_id,event_id,skill_id,version,digest,
                action,created_at FROM skill_state_audit WHERE workspace_id=?
                ORDER BY created_at DESC,event_id DESC LIMIT ?""", (workspace, limit)).fetchall()
        return [dict(row) for row in rows]
