"""Owner-scoped, explicit approval of a single semantic-preference write."""

from contextlib import contextmanager
import json
from pathlib import Path
import re
import time
from uuid import uuid4

from domain.events import EventType
from memory.semantic_memory import SemanticMemoryRepository, validate_preference
from storage.conversations import ConversationRepository
from storage.workspaces import workspace_key


class ToolApprovalNotFound(Exception):
    """Approval is missing or outside the authenticated owner scope."""


class ToolApprovalConflict(Exception):
    """Approval is no longer pending or its turn changed."""


class ToolApprovalRepository:
    MAX_PENDING = 20
    RETENTION_SECONDS = 7 * 24 * 60 * 60

    def __init__(self, root: Path):
        self.root = Path(root)
        self.semantic = SemanticMemoryRepository(root)

    @contextmanager
    def _connect(self):
        # The semantic table and the turn/event/approval records share one SQLite file.
        with self.semantic._connect() as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS tool_approvals (
                    workspace_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                    approval_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL, capability_id TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','approved','rejected','expired')),
                    memory_id TEXT, created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL, expires_at INTEGER NOT NULL,
                    PRIMARY KEY(workspace_id,principal_id,approval_id),
                    FOREIGN KEY(workspace_id,principal_id,conversation_id,turn_id)
                      REFERENCES turns(workspace_id,principal_id,conversation_id,idempotency_key)
                      ON DELETE CASCADE);
                CREATE INDEX IF NOT EXISTS tool_approvals_pending
                  ON tool_approvals(workspace_id,principal_id,status,expires_at);
            """)
            yield connection

    @staticmethod
    def _validate_owner(workspace: str, principal: str, approval_id: str | None = None) -> None:
        workspace_key(workspace)
        if not principal or len(principal) > 256:
            raise ValueError("Invalid principal")
        if approval_id is not None and not re.fullmatch(r"[a-f0-9]{32}", approval_id):
            raise ToolApprovalNotFound()

    def propose_preference(self, workspace: str, principal: str, conversation_id: str,
                           turn_id: str, lease_token: str, key: str, value: str,
                           *, now: int | None = None) -> dict:
        self._validate_owner(workspace, principal)
        ConversationRepository._validate(workspace, principal, conversation_id)
        display_key, memory_value = validate_preference(key, value)
        current = int(time.time() if now is None else now)
        approval_id = uuid4().hex
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            turn = connection.execute("""SELECT status,lease_token,lease_expires_at FROM turns
                WHERE workspace_id=? AND principal_id=? AND conversation_id=? AND idempotency_key=?""",
                (workspace, principal, conversation_id, turn_id)).fetchone()
            if turn is None or turn["status"] != "running" or turn["lease_token"] != lease_token \
                    or turn["lease_expires_at"] is None or turn["lease_expires_at"] <= current:
                raise ToolApprovalConflict()
            pending = connection.execute("""SELECT COUNT(*) FROM tool_approvals WHERE workspace_id=?
                AND principal_id=? AND status='pending' AND expires_at>?""",
                (workspace, principal, current)).fetchone()[0]
            if pending >= self.MAX_PENDING:
                raise ToolApprovalConflict()
            arguments = json.dumps({"key": display_key, "value": memory_value},
                                   ensure_ascii=False, separators=(",", ":"))
            connection.execute("""INSERT INTO tool_approvals VALUES (
                ?,?,?,?,?,? ,?,'pending',NULL,?,?,?)""",
                (workspace, principal, approval_id, conversation_id, turn_id,
                 "memory.save_preference", arguments, current, current,
                 current + self.RETENTION_SECONDS))
            ConversationRepository._append_event(
                connection, workspace, principal, conversation_id, turn_id,
                EventType.APPROVAL_REQUIRED,
                {"approval_id": approval_id, "capability_id": "memory.save_preference",
                 "status": "pending"}, f"approval.required:{approval_id}", current,
            )
            connection.commit()
        return {"approval_id": approval_id, "capability_id": "memory.save_preference",
                "status": "pending", "key": display_key, "value": memory_value,
                "expires_at": current + self.RETENTION_SECONDS}

    def resolve(self, workspace: str, principal: str, approval_id: str,
                decision: str, *, now: int | None = None) -> dict:
        self._validate_owner(workspace, principal, approval_id)
        if decision not in ("approve", "reject"):
            raise ValueError("Invalid approval decision")
        current = int(time.time() if now is None else now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("""SELECT * FROM tool_approvals WHERE workspace_id=?
                AND principal_id=? AND approval_id=?""",
                (workspace, principal, approval_id)).fetchone()
            if row is None:
                raise ToolApprovalNotFound()
            if row["status"] != "pending":
                if row["status"] == ("approved" if decision == "approve" else "rejected"):
                    connection.commit()
                    return {"approval_id": approval_id, "status": row["status"],
                            "memory_id": row["memory_id"]}
                raise ToolApprovalConflict()
            turn = connection.execute("""SELECT status,response_json FROM turns WHERE workspace_id=?
                AND principal_id=? AND conversation_id=? AND idempotency_key=?""",
                (workspace, principal, row["conversation_id"], row["turn_id"])).fetchone()
            if turn is None or turn["status"] != "complete" or not turn["response_json"]:
                raise ToolApprovalConflict()
            response = json.loads(turn["response_json"])
            if response.get("approval", {}).get("approval_id") != approval_id:
                raise ToolApprovalConflict()
            status = "expired" if row["expires_at"] <= current else (
                "approved" if decision == "approve" else "rejected")
            memory_id = None
            if status == "approved":
                arguments = json.loads(row["arguments_json"])
                try:
                    saved = self.semantic.upsert_in_transaction(
                        connection, workspace, principal, "preference",
                        arguments["key"], arguments["value"], now=current,
                    )
                except ValueError as exc:
                    raise ToolApprovalConflict() from exc
                memory_id = saved["memory_id"]
            connection.execute("""UPDATE tool_approvals SET status=?,memory_id=?,updated_at=?
                WHERE workspace_id=? AND principal_id=? AND approval_id=? AND status='pending'""",
                (status, memory_id, current, workspace, principal, approval_id))
            response["approval"]["status"] = status
            response["response"] = {
                "approved": "Preference saved after your approval.",
                "rejected": "Preference was not saved.",
                "expired": "The preference request expired without a change.",
            }[status]
            response["result_status"] = f"approval_{status}"
            response["tool_result"] = ([{"capability_id": row["capability_id"],
                                        "memory_id": memory_id}] if memory_id else None)
            connection.execute("""UPDATE turns SET response_json=? WHERE workspace_id=?
                AND principal_id=? AND conversation_id=? AND idempotency_key=?""",
                (json.dumps(response, ensure_ascii=False, separators=(",", ":")),
                 workspace, principal, row["conversation_id"], row["turn_id"]))
            connection.execute("""UPDATE turn_events SET payload_json=? WHERE workspace_id=?
                AND principal_id=? AND conversation_id=? AND turn_id=?
                AND event_type='message.completed'""",
                (json.dumps({"response": response["response"],
                             "result_status": response["result_status"]}, separators=(",", ":")),
                 workspace, principal, row["conversation_id"], row["turn_id"]))
            ConversationRepository._append_event(
                connection, workspace, principal, row["conversation_id"], row["turn_id"],
                EventType.APPROVAL_RESOLVED,
                {"approval_id": approval_id, "status": status},
                f"approval.resolved:{approval_id}", current,
            )
            ConversationRepository._append_event(
                connection, workspace, principal, row["conversation_id"], row["turn_id"],
                EventType.TOOL_COMPLETED if status == "approved" else EventType.TOOL_FAILED,
                {"capability_id": row["capability_id"], "status": status},
                f"approval.tool:{approval_id}", current,
            )
            connection.execute("""DELETE FROM conversation_summaries WHERE workspace_id=?
                AND principal_id=? AND conversation_id=?""",
                (workspace, principal, row["conversation_id"]))
            ConversationRepository._refresh_summary(connection, workspace, principal,
                                                    row["conversation_id"])
            connection.commit()
        return {"approval_id": approval_id, "status": status, "memory_id": memory_id}

    def get(self, workspace: str, principal: str, approval_id: str) -> dict:
        self._validate_owner(workspace, principal, approval_id)
        with self._connect() as connection:
            row = connection.execute("""SELECT capability_id,arguments_json,status,memory_id,
                expires_at FROM tool_approvals WHERE workspace_id=? AND principal_id=?
                AND approval_id=?""", (workspace, principal, approval_id)).fetchone()
        if row is None:
            raise ToolApprovalNotFound()
        arguments = json.loads(row["arguments_json"])
        return {"approval_id": approval_id, "capability_id": row["capability_id"],
                "key": arguments["key"], "value": arguments["value"],
                "status": row["status"], "memory_id": row["memory_id"],
                "expires_at": row["expires_at"]}
