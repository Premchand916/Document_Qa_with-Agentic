"""Durable, owner-scoped ingestion jobs with expiring worker leases."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from uuid import uuid4


ACTIVE_STATUSES = ("queued", "running", "cancel_requested")
TERMINAL_STATUSES = ("ready", "failed", "cancelled")


class IngestionJobNotFound(Exception):
    """Job is absent or outside the authorized owner boundary."""


class IngestionJobConflict(Exception):
    """Idempotency key was reused with a different payload or stale lease."""


class IngestionCapacityError(Exception):
    """Workspace has reached its bounded pending-job limit."""


class IngestionCancelled(Exception):
    """Cooperative cancellation was observed before publication."""


def payload_fingerprint(files: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, content in files:
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


class IngestionJobRepository:
    def __init__(self, root: Path, *, lease_seconds: int = 120, retention_days: int = 7,
                 max_pending_per_workspace: int = 10):
        if not 10 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be between 10 and 3600")
        if not 1 <= retention_days <= 365:
            raise ValueError("retention_days must be between 1 and 365")
        if not 1 <= max_pending_per_workspace <= 100:
            raise ValueError("max_pending_per_workspace must be between 1 and 100")
        self.root = Path(root)
        self.lease_seconds = lease_seconds
        self.retention_days = retention_days
        self.max_pending_per_workspace = max_pending_per_workspace
        self._initialize()

    @contextmanager
    def _connect(self):
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        connection = sqlite3.connect(self.root / "ingestion_jobs.sqlite3", timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    workspace_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    expected_version TEXT,
                    status TEXT NOT NULL CHECK(status IN
                        ('queued','running','cancel_requested','ready','failed','cancelled')),
                    lease_token TEXT,
                    lease_expires_at INTEGER,
                    current_stage TEXT,
                    stages_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    http_status INTEGER,
                    created_at INTEGER NOT NULL,
                    started_at INTEGER,
                    finished_at INTEGER,
                    expires_at INTEGER NOT NULL,
                    PRIMARY KEY(workspace_id, principal_id, job_id),
                    UNIQUE(workspace_id, principal_id, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS ingestion_inputs (
                    workspace_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    content BLOB NOT NULL,
                    sha256 TEXT NOT NULL,
                    PRIMARY KEY(workspace_id, principal_id, job_id, position),
                    FOREIGN KEY(workspace_id, principal_id, job_id)
                        REFERENCES ingestion_jobs(workspace_id, principal_id, job_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS ingestion_jobs_status_idx
                    ON ingestion_jobs(status, lease_expires_at, created_at);
            """)
            connection.commit()

    @staticmethod
    def _now_ms(now: float | None = None) -> int:
        return int((time.time() if now is None else now) * 1000)

    def _public(self, row: sqlite3.Row) -> dict:
        stages = json.loads(row["stages_json"] or "{}")
        queue_ms = None if row["started_at"] is None else max(0, row["started_at"] - row["created_at"])
        end = row["finished_at"] or self._now_ms()
        run_ms = None if row["started_at"] is None else max(0, end - row["started_at"])
        return {
            "job_id": row["job_id"],
            "workspace_id": row["workspace_id"],
            "status": row["status"],
            "current_stage": row["current_stage"],
            "stage_durations_ms": stages,
            "queue_duration_ms": queue_ms,
            "run_duration_ms": run_ms,
            "created_at_ms": row["created_at"],
            "finished_at_ms": row["finished_at"],
            "error": None if row["error_code"] is None else {
                "code": row["error_code"],
                "message": row["error_message"],
                "http_status": row["http_status"],
            },
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
        }

    def submit(self, workspace: str, principal: str, idempotency_key: str,
               files: list[tuple[str, bytes]], expected_version: str | None,
               *, now: float | None = None) -> dict:
        fingerprint = payload_fingerprint(files)
        now_ms = self._now_ms(now)
        expires_at = now_ms + self.retention_days * 86_400_000
        job_id = uuid4().hex
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("""SELECT * FROM ingestion_jobs
                WHERE workspace_id=? AND principal_id=? AND idempotency_key=?""",
                (workspace, principal, idempotency_key)).fetchone()
            if existing is not None:
                if existing["payload_sha256"] != fingerprint:
                    raise IngestionJobConflict("Idempotency key payload mismatch")
                connection.commit()
                return self._public(existing)
            pending = connection.execute("""SELECT COUNT(*) FROM ingestion_jobs
                WHERE workspace_id=? AND status IN ('queued','running','cancel_requested')""",
                (workspace,)).fetchone()[0]
            if pending >= self.max_pending_per_workspace:
                raise IngestionCapacityError("Workspace ingestion capacity is full")
            connection.execute("""INSERT INTO ingestion_jobs(
                workspace_id,principal_id,job_id,idempotency_key,payload_sha256,expected_version,
                status,created_at,expires_at) VALUES (?,?,?,?,?,?,'queued',?,?)""",
                (workspace, principal, job_id, idempotency_key, fingerprint, expected_version,
                 now_ms, expires_at))
            connection.executemany("""INSERT INTO ingestion_inputs(
                workspace_id,principal_id,job_id,position,filename,content,sha256)
                VALUES (?,?,?,?,?,?,?)""", [
                    (workspace, principal, job_id, position, name, content,
                     hashlib.sha256(content).hexdigest())
                    for position, (name, content) in enumerate(files)
                ])
            connection.commit()
        return self.get(workspace, principal, job_id)

    def get(self, workspace: str, principal: str, job_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute("""SELECT * FROM ingestion_jobs
                WHERE workspace_id=? AND principal_id=? AND job_id=?""",
                (workspace, principal, job_id)).fetchone()
        if row is None:
            raise IngestionJobNotFound()
        return self._public(row)

    def list_recent(self, workspace: str, principal: str, limit: int = 20) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("""SELECT * FROM ingestion_jobs
                WHERE workspace_id=? AND principal_id=? ORDER BY created_at DESC LIMIT ?""",
                (workspace, principal, limit)).fetchall()
        return [self._public(row) for row in rows]

    def queued_ids(self, limit: int = 100) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute("""SELECT job_id FROM ingestion_jobs
                WHERE status='queued' ORDER BY created_at LIMIT ?""", (limit,)).fetchall()
        return [row["job_id"] for row in rows]

    def claim(self, job_id: str, *, now: float | None = None) -> str | None:
        now_ms = self._now_ms(now)
        lease_token = uuid4().hex
        lease_expiry = now_ms + self.lease_seconds * 1000
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM ingestion_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None or row["status"] in TERMINAL_STATUSES:
                connection.commit()
                return None
            if row["status"] == "cancel_requested":
                self._finish_cancelled(connection, job_id, now_ms)
                connection.commit()
                return None
            if row["status"] == "running" and (row["lease_expires_at"] or 0) > now_ms:
                connection.commit()
                return None
            updated = connection.execute("""UPDATE ingestion_jobs SET status='running',
                lease_token=?,lease_expires_at=?,started_at=COALESCE(started_at,?),current_stage='queued'
                WHERE job_id=? AND (status='queued' OR (status='running' AND lease_expires_at<=?))""",
                (lease_token, lease_expiry, now_ms, job_id, now_ms)).rowcount
            connection.commit()
        return lease_token if updated == 1 else None

    def renew(self, job_id: str, lease_token: str, *, now: float | None = None) -> None:
        now_ms = self._now_ms(now)
        with self._connect() as connection:
            updated = connection.execute("""UPDATE ingestion_jobs SET lease_expires_at=?
                WHERE job_id=? AND lease_token=? AND status IN ('running','cancel_requested')
                AND lease_expires_at>?""",
                (now_ms + self.lease_seconds * 1000, job_id, lease_token, now_ms)).rowcount
            connection.commit()
        if updated != 1:
            raise IngestionJobConflict("Ingestion lease was lost")

    def load_inputs(self, job_id: str, lease_token: str) -> tuple[dict, list[tuple[str, bytes]]]:
        now_ms = self._now_ms()
        with self._connect() as connection:
            row = connection.execute("""SELECT * FROM ingestion_jobs
                WHERE job_id=? AND lease_token=? AND status IN ('running','cancel_requested')
                AND lease_expires_at>?""", (job_id, lease_token, now_ms)).fetchone()
            if row is None:
                raise IngestionJobConflict("Ingestion lease was lost")
            inputs = connection.execute("""SELECT filename,content FROM ingestion_inputs
                WHERE workspace_id=? AND principal_id=? AND job_id=? ORDER BY position""",
                (row["workspace_id"], row["principal_id"], job_id)).fetchall()
        return dict(row), [(item["filename"], bytes(item["content"])) for item in inputs]

    def check_cancelled(self, job_id: str, lease_token: str) -> None:
        now_ms = self._now_ms()
        with self._connect() as connection:
            row = connection.execute("""SELECT status,lease_token,lease_expires_at
                FROM ingestion_jobs WHERE job_id=?""",
                                     (job_id,)).fetchone()
        if row is None or row["lease_token"] != lease_token or (row["lease_expires_at"] or 0) <= now_ms:
            raise IngestionJobConflict("Ingestion lease was lost")
        if row["status"] == "cancel_requested":
            raise IngestionCancelled()

    def set_stage(self, job_id: str, lease_token: str, stage: str) -> None:
        now_ms = self._now_ms()
        with self._connect() as connection:
            updated = connection.execute("""UPDATE ingestion_jobs SET current_stage=?
                WHERE job_id=? AND lease_token=? AND status IN ('running','cancel_requested')
                AND lease_expires_at>?""", (stage, job_id, lease_token, now_ms)).rowcount
            connection.commit()
        if updated != 1:
            raise IngestionJobConflict("Ingestion lease was lost")

    def record_stage(self, job_id: str, lease_token: str, stage: str, duration_ms: int) -> None:
        now_ms = self._now_ms()
        with self._connect() as connection:
            row = connection.execute("""SELECT stages_json FROM ingestion_jobs
                WHERE job_id=? AND lease_token=? AND status IN ('running','cancel_requested')
                AND lease_expires_at>?""", (job_id, lease_token, now_ms)).fetchone()
            if row is None:
                raise IngestionJobConflict("Ingestion lease was lost")
            stages = json.loads(row["stages_json"] or "{}")
            stages[stage] = max(0, int(duration_ms))
            connection.execute("UPDATE ingestion_jobs SET stages_json=? WHERE job_id=?",
                               (json.dumps(stages, separators=(",", ":")), job_id))
            connection.commit()

    def complete(self, job_id: str, lease_token: str, result: dict,
                 *, now: float | None = None) -> None:
        now_ms = self._now_ms(now)
        payload = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute("""UPDATE ingestion_jobs SET status='ready',result_json=?,
                current_stage='ready',finished_at=?,lease_token=NULL,lease_expires_at=NULL
                WHERE job_id=? AND lease_token=? AND status IN ('running','cancel_requested')
                AND lease_expires_at>?""", (payload, now_ms, job_id, lease_token, now_ms)).rowcount
            if updated != 1:
                raise IngestionJobConflict("Ingestion lease was lost")
            connection.execute("DELETE FROM ingestion_inputs WHERE job_id=?", (job_id,))
            connection.commit()

    def fail(self, job_id: str, lease_token: str, code: str, message: str, http_status: int,
             *, now: float | None = None) -> None:
        now_ms = self._now_ms(now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute("""UPDATE ingestion_jobs SET status='failed',error_code=?,
                error_message=?,http_status=?,current_stage='failed',finished_at=?,
                lease_token=NULL,lease_expires_at=NULL
                WHERE job_id=? AND lease_token=? AND status IN ('running','cancel_requested')
                AND lease_expires_at>?""",
                (code, message, http_status, now_ms, job_id, lease_token, now_ms)).rowcount
            if updated != 1:
                raise IngestionJobConflict("Ingestion lease was lost")
            connection.execute("DELETE FROM ingestion_inputs WHERE job_id=?", (job_id,))
            connection.commit()

    def _finish_cancelled(self, connection: sqlite3.Connection, job_id: str, now_ms: int) -> None:
        connection.execute("""UPDATE ingestion_jobs SET status='cancelled',current_stage='cancelled',
            finished_at=?,lease_token=NULL,lease_expires_at=NULL WHERE job_id=?""", (now_ms, job_id))
        connection.execute("DELETE FROM ingestion_inputs WHERE job_id=?", (job_id,))

    def finish_cancelled(self, job_id: str, lease_token: str, *, now: float | None = None) -> None:
        now_ms = self._now_ms(now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("""SELECT status,lease_token FROM ingestion_jobs
                WHERE job_id=? AND lease_expires_at>?""", (job_id, now_ms)).fetchone()
            if row is None or row["lease_token"] != lease_token or row["status"] not in ("running", "cancel_requested"):
                raise IngestionJobConflict("Ingestion lease was lost")
            self._finish_cancelled(connection, job_id, now_ms)
            connection.commit()

    def request_cancel(self, workspace: str, principal: str, job_id: str,
                       *, now: float | None = None) -> dict:
        now_ms = self._now_ms(now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("""SELECT * FROM ingestion_jobs
                WHERE workspace_id=? AND principal_id=? AND job_id=?""",
                (workspace, principal, job_id)).fetchone()
            if row is None:
                raise IngestionJobNotFound()
            if row["status"] == "queued":
                self._finish_cancelled(connection, job_id, now_ms)
            elif row["status"] == "running":
                connection.execute("UPDATE ingestion_jobs SET status='cancel_requested' WHERE job_id=?",
                                   (job_id,))
            connection.commit()
        return self.get(workspace, principal, job_id)

    def recover_expired(self, *, now: float | None = None) -> int:
        now_ms = self._now_ms(now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cancelled = connection.execute("""SELECT job_id FROM ingestion_jobs
                WHERE status='cancel_requested' AND lease_expires_at<=?""", (now_ms,)).fetchall()
            for row in cancelled:
                self._finish_cancelled(connection, row["job_id"], now_ms)
            recovered = connection.execute("""UPDATE ingestion_jobs SET status='queued',
                lease_token=NULL,lease_expires_at=NULL,current_stage='queued'
                WHERE status='running' AND lease_expires_at<=?""", (now_ms,)).rowcount
            connection.commit()
        return recovered

    def purge_expired(self, *, now: float | None = None) -> int:
        now_ms = self._now_ms(now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            deleted = connection.execute("""DELETE FROM ingestion_jobs
                WHERE status IN ('ready','failed','cancelled') AND expires_at<=?""", (now_ms,)).rowcount
            connection.commit()
        return deleted
