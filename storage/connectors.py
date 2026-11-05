"""Owner-scoped connector grants, sources, sync runs, and immutable snapshots."""

from contextlib import contextmanager
from pathlib import Path
import re
import sqlite3
import time
from uuid import uuid4

from domain.connectors import (ConnectorScan, ConnectorState, ConnectorType, SyncState,
                               validate_display_name, validate_source_key)
from storage.workspaces import workspace_key


class ConnectorNotFound(Exception):
    """Connector object is absent or outside owner scope."""


class ConnectorConflict(Exception):
    """Idempotency or lifecycle state conflicts with requested operation."""


class ConnectorRevoked(Exception):
    """Grant was revoked before operation completed."""


class ConnectorCapacityError(Exception):
    """Owner-scoped durable connector storage reached its configured bound."""


class ConnectorRepository:
    MAX_SOURCES_PER_OWNER = 10
    MAX_SNAPSHOTS_PER_SOURCE = 10
    RETAIN_SYNC_RUNS_PER_SOURCE = 200

    def __init__(self, root: Path):
        self.root = Path(root)
        self._initialize()

    @contextmanager
    def _connect(self):
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        path = self.root / "connectors.sqlite3"
        path.touch(mode=0o600, exist_ok=True)
        connection = sqlite3.connect(path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=10000")
            connection.execute("PRAGMA synchronous=FULL")
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS connector_grants (
                    workspace_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                    grant_id TEXT NOT NULL, connector_type TEXT NOT NULL
                      CHECK(connector_type IN ('managed_folder')),
                    state TEXT NOT NULL CHECK(state IN ('active','revoked')),
                    created_at INTEGER NOT NULL, revoked_at INTEGER,
                    PRIMARY KEY(workspace_id,principal_id,grant_id));
                CREATE TABLE IF NOT EXISTS connector_owner_epochs (
                    workspace_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                    revocation_epoch INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(workspace_id,principal_id));
                CREATE TABLE IF NOT EXISTS connector_sources (
                    workspace_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                    source_id TEXT NOT NULL, grant_id TEXT NOT NULL,
                    source_key TEXT NOT NULL, display_name TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL, state TEXT NOT NULL
                      CHECK(state IN ('active','revoked')),
                    last_cursor TEXT, last_snapshot_id TEXT, last_synced_at INTEGER,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY(workspace_id,principal_id,source_id),
                    UNIQUE(workspace_id,principal_id,idempotency_key),
                    UNIQUE(workspace_id,principal_id,source_key),
                    FOREIGN KEY(workspace_id,principal_id,grant_id)
                      REFERENCES connector_grants(workspace_id,principal_id,grant_id));
                CREATE TABLE IF NOT EXISTS connector_sync_runs (
                    workspace_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                    run_id TEXT NOT NULL, source_id TEXT NOT NULL, idempotency_key TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('running','ready','failed')),
                    cursor_before TEXT, cursor_after TEXT, snapshot_id TEXT, changed INTEGER,
                    error_code TEXT, created_at INTEGER NOT NULL, finished_at INTEGER,
                    PRIMARY KEY(workspace_id,principal_id,run_id),
                    UNIQUE(workspace_id,principal_id,idempotency_key),
                    FOREIGN KEY(workspace_id,principal_id,source_id)
                      REFERENCES connector_sources(workspace_id,principal_id,source_id));
                CREATE TABLE IF NOT EXISTS connector_snapshots (
                    workspace_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                    snapshot_id TEXT NOT NULL, source_id TEXT NOT NULL,
                    cursor TEXT NOT NULL, file_count INTEGER NOT NULL,
                    total_bytes INTEGER NOT NULL, created_at INTEGER NOT NULL,
                    PRIMARY KEY(workspace_id,principal_id,snapshot_id),
                    UNIQUE(workspace_id,principal_id,source_id,cursor),
                    FOREIGN KEY(workspace_id,principal_id,source_id)
                      REFERENCES connector_sources(workspace_id,principal_id,source_id));
                CREATE TABLE IF NOT EXISTS connector_snapshot_files (
                    workspace_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                    snapshot_id TEXT NOT NULL, relative_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL, media_type TEXT NOT NULL, byte_count INTEGER NOT NULL,
                    content BLOB NOT NULL,
                    PRIMARY KEY(workspace_id,principal_id,snapshot_id,relative_path),
                    FOREIGN KEY(workspace_id,principal_id,snapshot_id)
                      REFERENCES connector_snapshots(workspace_id,principal_id,snapshot_id)
                      ON DELETE CASCADE);
                CREATE INDEX IF NOT EXISTS connector_sources_recent
                    ON connector_sources(workspace_id,principal_id,created_at);
                CREATE INDEX IF NOT EXISTS connector_sync_recent
                    ON connector_sync_runs(workspace_id,principal_id,source_id,created_at);
            """)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(connector_sources)")}
            if "published_snapshot_id" not in columns:
                connection.execute("ALTER TABLE connector_sources ADD COLUMN published_snapshot_id TEXT")
            if "published_file_count" not in columns:
                connection.execute("ALTER TABLE connector_sources ADD COLUMN published_file_count INTEGER")
            connection.commit()

    @staticmethod
    def _validate_owner(workspace: str, principal: str) -> None:
        workspace_key(workspace)
        if not principal or len(principal) > 256:
            raise ValueError("Invalid principal")

    @staticmethod
    def _validate_id(value: str, label: str) -> None:
        if not re.fullmatch(r"[a-f0-9]{32}", value):
            raise ValueError(f"Invalid {label}")

    @staticmethod
    def _validate_key(value: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{8,128}", value):
            raise ValueError("Invalid idempotency key")

    @staticmethod
    def _source(row: sqlite3.Row) -> dict:
        return {
            "source_id": row["source_id"], "grant_id": row["grant_id"],
            "connector_type": row["connector_type"], "source_key": row["source_key"],
            "display_name": row["display_name"], "state": row["source_state"],
            "grant_state": row["grant_state"], "last_cursor": row["last_cursor"],
            "last_snapshot_id": row["last_snapshot_id"],
            "published_snapshot_id": row["published_snapshot_id"],
            "published_file_count": row["published_file_count"],
            "last_synced_at": row["last_synced_at"], "created_at": row["created_at"],
        }

    @staticmethod
    def _run(row: sqlite3.Row) -> dict:
        return {
            "run_id": row["run_id"], "source_id": row["source_id"],
            "status": row["status"], "cursor_before": row["cursor_before"],
            "cursor_after": row["cursor_after"], "snapshot_id": row["snapshot_id"],
            "changed": None if row["changed"] is None else bool(row["changed"]),
            "error_code": row["error_code"], "created_at": row["created_at"],
            "finished_at": row["finished_at"],
        }

    def create_source(self, workspace: str, principal: str, source_key: str,
                      display_name: str, idempotency_key: str,
                      *, now: int | None = None) -> dict:
        self._validate_owner(workspace, principal)
        source_key = validate_source_key(source_key)
        display_name = validate_display_name(display_name)
        self._validate_key(idempotency_key)
        current_time = int(time.time() if now is None else now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("""SELECT s.*,g.connector_type,g.state AS grant_state,
                s.state AS source_state FROM connector_sources s JOIN connector_grants g
                ON g.workspace_id=s.workspace_id AND g.principal_id=s.principal_id
                AND g.grant_id=s.grant_id WHERE s.workspace_id=? AND s.principal_id=?
                AND s.idempotency_key=?""", (workspace, principal, idempotency_key)).fetchone()
            if existing is not None:
                if existing["source_key"] != source_key or existing["display_name"] != display_name:
                    raise ConnectorConflict()
                connection.commit()
                return self._source(existing)
            collision = connection.execute("""SELECT 1 FROM connector_sources
                WHERE workspace_id=? AND principal_id=? AND source_key=?""",
                (workspace, principal, source_key)).fetchone()
            if collision is not None:
                raise ConnectorConflict()
            source_count = connection.execute("""SELECT COUNT(*) FROM connector_sources
                WHERE workspace_id=? AND principal_id=?""",
                (workspace, principal)).fetchone()[0]
            if source_count >= self.MAX_SOURCES_PER_OWNER:
                raise ConnectorCapacityError()
            grant_id, source_id = uuid4().hex, uuid4().hex
            connection.execute("""INSERT INTO connector_grants(
                workspace_id,principal_id,grant_id,connector_type,state,created_at)
                VALUES (?,?,?,?,'active',?)""",
                (workspace, principal, grant_id, ConnectorType.MANAGED_FOLDER.value, current_time))
            connection.execute("""INSERT INTO connector_sources(
                workspace_id,principal_id,source_id,grant_id,source_key,display_name,
                idempotency_key,state,created_at) VALUES (?,?,?,?,?,?,?,'active',?)""",
                (workspace, principal, source_id, grant_id, source_key, display_name,
                 idempotency_key, current_time))
            connection.commit()
        return self.get_source(workspace, principal, source_id)

    def get_source(self, workspace: str, principal: str, source_id: str) -> dict:
        self._validate_owner(workspace, principal)
        self._validate_id(source_id, "source identifier")
        with self._connect() as connection:
            row = connection.execute("""SELECT s.*,g.connector_type,g.state AS grant_state,
                s.state AS source_state FROM connector_sources s JOIN connector_grants g
                ON g.workspace_id=s.workspace_id AND g.principal_id=s.principal_id
                AND g.grant_id=s.grant_id WHERE s.workspace_id=? AND s.principal_id=?
                AND s.source_id=?""", (workspace, principal, source_id)).fetchone()
        if row is None:
            raise ConnectorNotFound()
        return self._source(row)

    def list_sources(self, workspace: str, principal: str) -> list[dict]:
        self._validate_owner(workspace, principal)
        with self._connect() as connection:
            rows = connection.execute("""SELECT s.*,g.connector_type,g.state AS grant_state,
                s.state AS source_state FROM connector_sources s JOIN connector_grants g
                ON g.workspace_id=s.workspace_id AND g.principal_id=s.principal_id
                AND g.grant_id=s.grant_id WHERE s.workspace_id=? AND s.principal_id=?
                ORDER BY s.created_at DESC,s.source_id DESC""", (workspace, principal)).fetchall()
        return [self._source(row) for row in rows]

    def mark_published(self, workspace: str, principal: str, source_id: str,
                       snapshot_id: str) -> dict:
        self._validate_owner(workspace, principal)
        self._validate_id(source_id, "source identifier")
        self._validate_id(snapshot_id, "snapshot identifier")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            snapshot = connection.execute("""SELECT file_count FROM connector_snapshots
                WHERE workspace_id=? AND principal_id=? AND source_id=? AND snapshot_id=?""",
                (workspace, principal, source_id, snapshot_id)).fetchone()
            if snapshot is None:
                raise ConnectorConflict()
            updated = connection.execute("""UPDATE connector_sources SET published_snapshot_id=?,
                published_file_count=?
                WHERE workspace_id=? AND principal_id=? AND source_id=?
                AND state='active' AND last_snapshot_id=? AND EXISTS (
                  SELECT 1 FROM connector_grants g WHERE g.workspace_id=connector_sources.workspace_id
                  AND g.principal_id=connector_sources.principal_id
                  AND g.grant_id=connector_sources.grant_id AND g.state='active')""",
                (snapshot_id, snapshot["file_count"], workspace, principal, source_id, snapshot_id))
            if updated.rowcount != 1:
                raise ConnectorConflict()
            connection.commit()
        return self.get_source(workspace, principal, source_id)

    def assert_published_current(self, workspace: str, principal: str,
                                 source_ids: list[str]) -> None:
        self._validate_owner(workspace, principal)
        if not source_ids:
            return
        if len(source_ids) > self.MAX_SOURCES_PER_OWNER:
            raise ConnectorConflict()
        with self._connect() as connection:
            for source_id in source_ids:
                self._validate_id(source_id, "source identifier")
                row = connection.execute("""SELECT s.state,s.last_snapshot_id,
                    s.published_snapshot_id,g.state AS grant_state
                    FROM connector_sources s JOIN connector_grants g
                    ON g.workspace_id=s.workspace_id AND g.principal_id=s.principal_id
                    AND g.grant_id=s.grant_id WHERE s.workspace_id=? AND s.principal_id=?
                    AND s.source_id=?""", (workspace, principal, source_id)).fetchone()
                if (row is None or row["state"] != "active" or row["grant_state"] != "active"
                        or row["published_snapshot_id"] is None
                        or row["published_snapshot_id"] != row["last_snapshot_id"]):
                    raise ConnectorRevoked()

    def revoke_grant(self, workspace: str, principal: str, grant_id: str,
                     *, now: int | None = None) -> dict:
        self._validate_owner(workspace, principal)
        self._validate_id(grant_id, "grant identifier")
        current_time = int(time.time() if now is None else now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("""SELECT state FROM connector_grants
                WHERE workspace_id=? AND principal_id=? AND grant_id=?""",
                (workspace, principal, grant_id)).fetchone()
            if row is None:
                raise ConnectorNotFound()
            if row["state"] == "active":
                connection.execute("""INSERT INTO connector_owner_epochs(
                    workspace_id,principal_id,revocation_epoch) VALUES (?,?,1)
                    ON CONFLICT(workspace_id,principal_id) DO UPDATE SET
                    revocation_epoch=revocation_epoch+1""", (workspace, principal))
            connection.execute("""UPDATE connector_grants SET state='revoked',
                revoked_at=COALESCE(revoked_at,?) WHERE workspace_id=? AND principal_id=?
                AND grant_id=?""", (current_time, workspace, principal, grant_id))
            connection.execute("""UPDATE connector_sources SET state='revoked'
                WHERE workspace_id=? AND principal_id=? AND grant_id=?""",
                (workspace, principal, grant_id))
            connection.commit()
        return {"grant_id": grant_id, "state": ConnectorState.REVOKED.value}

    def revocation_epoch(self, workspace: str, principal: str) -> int:
        self._validate_owner(workspace, principal)
        with self._connect() as connection:
            row = connection.execute("""SELECT revocation_epoch FROM connector_owner_epochs
                WHERE workspace_id=? AND principal_id=?""",
                (workspace, principal)).fetchone()
        return row["revocation_epoch"] if row else 0

    def delete_source(self, workspace: str, principal: str, source_id: str) -> dict:
        """Permanently remove a source, its copied content, and its dedicated grant."""
        self._validate_owner(workspace, principal)
        self._validate_id(source_id, "source identifier")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            source = connection.execute("""SELECT grant_id FROM connector_sources
                WHERE workspace_id=? AND principal_id=? AND source_id=?""",
                (workspace, principal, source_id)).fetchone()
            if source is None:
                raise ConnectorNotFound()
            snapshot_ids = connection.execute("""SELECT snapshot_id FROM connector_snapshots
                WHERE workspace_id=? AND principal_id=? AND source_id=?""",
                (workspace, principal, source_id)).fetchall()
            connection.executemany("""DELETE FROM connector_snapshot_files
                WHERE workspace_id=? AND principal_id=? AND snapshot_id=?""",
                [(workspace, principal, row["snapshot_id"]) for row in snapshot_ids])
            connection.execute("""DELETE FROM connector_snapshots
                WHERE workspace_id=? AND principal_id=? AND source_id=?""",
                (workspace, principal, source_id))
            connection.execute("""DELETE FROM connector_sync_runs
                WHERE workspace_id=? AND principal_id=? AND source_id=?""",
                (workspace, principal, source_id))
            connection.execute("""DELETE FROM connector_sources
                WHERE workspace_id=? AND principal_id=? AND source_id=?""",
                (workspace, principal, source_id))
            connection.execute("""DELETE FROM connector_grants
                WHERE workspace_id=? AND principal_id=? AND grant_id=?
                AND NOT EXISTS (SELECT 1 FROM connector_sources WHERE workspace_id=?
                    AND principal_id=? AND grant_id=?)""",
                (workspace, principal, source["grant_id"], workspace, principal,
                 source["grant_id"]))
            connection.commit()
        return {"source_id": source_id, "deleted": True}

    def begin_sync(self, workspace: str, principal: str, source_id: str,
                   idempotency_key: str, *, now: int | None = None) -> tuple[dict, bool]:
        self._validate_owner(workspace, principal)
        self._validate_id(source_id, "source identifier")
        self._validate_key(idempotency_key)
        current_time = int(time.time() if now is None else now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            source = connection.execute("""SELECT s.last_cursor,s.state AS source_state,
                g.state AS grant_state FROM connector_sources s JOIN connector_grants g
                ON g.workspace_id=s.workspace_id AND g.principal_id=s.principal_id
                AND g.grant_id=s.grant_id WHERE s.workspace_id=? AND s.principal_id=?
                AND s.source_id=?""", (workspace, principal, source_id)).fetchone()
            if source is None:
                raise ConnectorNotFound()
            if source["source_state"] != "active" or source["grant_state"] != "active":
                raise ConnectorRevoked()
            existing = connection.execute("""SELECT * FROM connector_sync_runs
                WHERE workspace_id=? AND principal_id=? AND idempotency_key=?""",
                (workspace, principal, idempotency_key)).fetchone()
            if existing is not None:
                if existing["source_id"] != source_id:
                    raise ConnectorConflict()
                connection.commit()
                return self._run(existing), False
            completed = connection.execute("""SELECT run_id FROM connector_sync_runs
                WHERE workspace_id=? AND principal_id=? AND source_id=? AND status!='running'
                ORDER BY created_at DESC,run_id DESC""",
                (workspace, principal, source_id)).fetchall()
            stale = completed[self.RETAIN_SYNC_RUNS_PER_SOURCE - 1:]
            connection.executemany("""DELETE FROM connector_sync_runs
                WHERE workspace_id=? AND principal_id=? AND run_id=?""",
                [(workspace, principal, row["run_id"]) for row in stale])
            running = connection.execute("""SELECT 1 FROM connector_sync_runs
                WHERE workspace_id=? AND principal_id=? AND source_id=? AND status='running'""",
                (workspace, principal, source_id)).fetchone()
            if running is not None:
                raise ConnectorConflict()
            run_id = uuid4().hex
            connection.execute("""INSERT INTO connector_sync_runs(
                workspace_id,principal_id,run_id,source_id,idempotency_key,status,
                cursor_before,created_at) VALUES (?,?,?,?,?,'running',?,?)""",
                (workspace, principal, run_id, source_id, idempotency_key,
                 source["last_cursor"], current_time))
            connection.commit()
        return self.get_sync(workspace, principal, run_id), True

    def complete_sync(self, workspace: str, principal: str, run_id: str,
                      scan: ConnectorScan, *, now: int | None = None) -> dict:
        self._validate_owner(workspace, principal)
        self._validate_id(run_id, "sync run identifier")
        current_time = int(time.time() if now is None else now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute("""SELECT r.*,s.source_key,s.state AS source_state,
                s.last_cursor,s.last_snapshot_id,g.state AS grant_state
                FROM connector_sync_runs r JOIN connector_sources s
                ON s.workspace_id=r.workspace_id AND s.principal_id=r.principal_id
                AND s.source_id=r.source_id JOIN connector_grants g
                ON g.workspace_id=s.workspace_id AND g.principal_id=s.principal_id
                AND g.grant_id=s.grant_id WHERE r.workspace_id=? AND r.principal_id=?
                AND r.run_id=?""", (workspace, principal, run_id)).fetchone()
            if run is None:
                raise ConnectorNotFound()
            if run["status"] != SyncState.RUNNING.value:
                raise ConnectorConflict()
            if run["source_state"] != "active" or run["grant_state"] != "active":
                raise ConnectorRevoked()
            if run["source_key"] != scan.source_key:
                raise ConnectorConflict()
            changed = scan.cursor != run["last_cursor"] or run["last_snapshot_id"] is None
            snapshot_id = run["last_snapshot_id"]
            if changed:
                prior = connection.execute("""SELECT snapshot_id FROM connector_snapshots
                    WHERE workspace_id=? AND principal_id=? AND source_id=? AND cursor=?""",
                    (workspace, principal, run["source_id"], scan.cursor)).fetchone()
                if prior is not None:
                    snapshot_id = prior["snapshot_id"]
                else:
                    snapshot_count = connection.execute("""SELECT COUNT(*)
                        FROM connector_snapshots WHERE workspace_id=? AND principal_id=?
                        AND source_id=?""",
                        (workspace, principal, run["source_id"])).fetchone()[0]
                    if snapshot_count >= self.MAX_SNAPSHOTS_PER_SOURCE:
                        raise ConnectorCapacityError()
                    snapshot_id = uuid4().hex
                    total = sum(len(item.content) for item in scan.files)
                    connection.execute("""INSERT INTO connector_snapshots(
                        workspace_id,principal_id,snapshot_id,source_id,cursor,file_count,
                        total_bytes,created_at) VALUES (?,?,?,?,?,?,?,?)""",
                        (workspace, principal, snapshot_id, run["source_id"], scan.cursor,
                         len(scan.files), total, current_time))
                    connection.executemany("""INSERT INTO connector_snapshot_files(
                        workspace_id,principal_id,snapshot_id,relative_path,sha256,media_type,
                        byte_count,content) VALUES (?,?,?,?,?,?,?,?)""", [
                        (workspace, principal, snapshot_id, item.relative_path, item.sha256,
                         item.media_type, len(item.content), item.content) for item in scan.files
                    ])
                connection.execute("""UPDATE connector_sources SET last_cursor=?,
                    last_snapshot_id=?,last_synced_at=? WHERE workspace_id=? AND principal_id=?
                    AND source_id=?""", (scan.cursor, snapshot_id, current_time,
                                          workspace, principal, run["source_id"]))
            else:
                connection.execute("""UPDATE connector_sources SET last_synced_at=?
                    WHERE workspace_id=? AND principal_id=? AND source_id=?""",
                    (current_time, workspace, principal, run["source_id"]))
            connection.execute("""UPDATE connector_sync_runs SET status='ready',cursor_after=?,
                snapshot_id=?,changed=?,finished_at=? WHERE workspace_id=? AND principal_id=?
                AND run_id=?""", (scan.cursor, snapshot_id, int(changed), current_time,
                                    workspace, principal, run_id))
            connection.commit()
        return self.get_sync(workspace, principal, run_id)

    def fail_sync(self, workspace: str, principal: str, run_id: str, error_code: str,
                  *, now: int | None = None) -> dict:
        self._validate_owner(workspace, principal)
        self._validate_id(run_id, "sync run identifier")
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", error_code):
            raise ValueError("Invalid connector error code")
        current_time = int(time.time() if now is None else now)
        with self._connect() as connection:
            updated = connection.execute("""UPDATE connector_sync_runs SET status='failed',
                error_code=?,finished_at=? WHERE workspace_id=? AND principal_id=?
                AND run_id=? AND status='running'""",
                (error_code, current_time, workspace, principal, run_id))
            connection.commit()
            if updated.rowcount != 1:
                raise ConnectorConflict()
        return self.get_sync(workspace, principal, run_id)

    def get_sync(self, workspace: str, principal: str, run_id: str) -> dict:
        self._validate_owner(workspace, principal)
        self._validate_id(run_id, "sync run identifier")
        with self._connect() as connection:
            row = connection.execute("""SELECT * FROM connector_sync_runs
                WHERE workspace_id=? AND principal_id=? AND run_id=?""",
                (workspace, principal, run_id)).fetchone()
        if row is None:
            raise ConnectorNotFound()
        return self._run(row)

    def list_snapshots(self, workspace: str, principal: str, source_id: str,
                       *, limit: int = 20) -> list[dict]:
        self._validate_owner(workspace, principal)
        self._validate_id(source_id, "source identifier")
        if not 1 <= limit <= 100:
            raise ValueError("Snapshot limit must be between 1 and 100")
        with self._connect() as connection:
            source = connection.execute("""SELECT s.state AS source_state,g.state AS grant_state
                FROM connector_sources s JOIN connector_grants g
                ON g.workspace_id=s.workspace_id AND g.principal_id=s.principal_id
                AND g.grant_id=s.grant_id WHERE s.workspace_id=? AND s.principal_id=?
                AND s.source_id=?""", (workspace, principal, source_id)).fetchone()
            if source is None:
                raise ConnectorNotFound()
            if source["source_state"] != "active" or source["grant_state"] != "active":
                raise ConnectorRevoked()
            rows = connection.execute("""SELECT snapshot_id,cursor,file_count,total_bytes,created_at
                FROM connector_snapshots WHERE workspace_id=? AND principal_id=? AND source_id=?
                ORDER BY created_at DESC,snapshot_id DESC LIMIT ?""",
                (workspace, principal, source_id, limit)).fetchall()
            result = []
            for row in rows:
                files = connection.execute("""SELECT relative_path,sha256,media_type,byte_count
                    FROM connector_snapshot_files WHERE workspace_id=? AND principal_id=?
                    AND snapshot_id=? ORDER BY relative_path""",
                    (workspace, principal, row["snapshot_id"])).fetchall()
                result.append({**dict(row), "source_id": source_id,
                               "files": [dict(item) for item in files]})
        return result

    def load_snapshot_files(self, workspace: str, principal: str,
                            snapshot_id: str) -> list[tuple[str, bytes, dict]]:
        self._validate_owner(workspace, principal)
        self._validate_id(snapshot_id, "snapshot identifier")
        with self._connect() as connection:
            snapshot = connection.execute("""SELECT s.source_id,cs.state AS source_state,
                g.state AS grant_state FROM connector_snapshots s JOIN connector_sources cs
                ON cs.workspace_id=s.workspace_id AND cs.principal_id=s.principal_id
                AND cs.source_id=s.source_id JOIN connector_grants g
                ON g.workspace_id=cs.workspace_id AND g.principal_id=cs.principal_id
                AND g.grant_id=cs.grant_id WHERE s.workspace_id=? AND s.principal_id=?
                AND s.snapshot_id=?""", (workspace, principal, snapshot_id)).fetchone()
            if snapshot is None:
                raise ConnectorNotFound()
            if snapshot["source_state"] != "active" or snapshot["grant_state"] != "active":
                raise ConnectorRevoked()
            rows = connection.execute("""SELECT relative_path,content,sha256,media_type
                FROM connector_snapshot_files WHERE workspace_id=? AND principal_id=?
                AND snapshot_id=? ORDER BY relative_path""",
                (workspace, principal, snapshot_id)).fetchall()
        return [(row["relative_path"], bytes(row["content"]),
                 {"sha256": row["sha256"], "media_type": row["media_type"],
                  "connector_snapshot_id": snapshot_id, "connector_source_id": snapshot["source_id"]})
                for row in rows]
