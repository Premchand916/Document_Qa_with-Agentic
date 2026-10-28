"""Backend-neutral memory snapshot contract and copy-only migration rehearsal."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from memory.episodic_memory import EpisodicMemoryRepository
from memory.semantic_memory import SemanticMemoryRepository
from storage.conversations import ConversationRepository


TABLE_COLUMNS = {
    "conversations": ("workspace_id", "principal_id", "conversation_id", "title", "created_at", "updated_at"),
    "turns": ("workspace_id", "principal_id", "conversation_id", "sequence", "idempotency_key", "question",
              "status", "response_json", "corpus_version", "lease_token", "lease_expires_at"),
    "conversation_summaries": ("workspace_id", "principal_id", "conversation_id", "through_sequence", "summary", "updated_at"),
    "semantic_memories": ("workspace_id", "principal_id", "memory_id", "category", "memory_key", "display_key",
                          "memory_value", "source", "created_at", "updated_at", "expires_at"),
    "episodes": ("workspace_id", "principal_id", "episode_id", "conversation_id", "idempotency_key", "task_key",
                 "correction", "validation_status", "corpus_version", "source_ids_json", "tool_context_json",
                 "evidence_json", "result_json", "created_at", "expires_at"),
    "episode_reviews": ("workspace_id", "principal_id", "episode_id", "review_id", "reviewer_id", "previous_status",
                        "decision", "review_note", "reviewed_at"),
}
TABLE_KEYS = {
    "conversations": ("workspace_id", "principal_id", "conversation_id"),
    "turns": ("workspace_id", "principal_id", "conversation_id", "sequence"),
    "conversation_summaries": ("workspace_id", "principal_id", "conversation_id"),
    "semantic_memories": ("workspace_id", "principal_id", "memory_id"),
    "episodes": ("workspace_id", "principal_id", "episode_id"),
    "episode_reviews": ("workspace_id", "principal_id", "episode_id", "review_id"),
}


class MemoryMigrationBlocked(RuntimeError):
    """Snapshot cannot be proven consistent or portable."""


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def _initialize(root: Path) -> Path:
    ConversationRepository(root).list_recent("migration", "initializer", limit=1)
    SemanticMemoryRepository(root).list_recent("migration", "initializer", limit=1)
    EpisodicMemoryRepository(root).list_recent("migration", "initializer", limit=1)
    return root / "conversations.sqlite3"


def validate_memory_snapshot(snapshot: dict) -> dict:
    if snapshot.get("schema_version") != 1 or snapshot.get("kind") != "docqa_memory_snapshot":
        raise ValueError("Unsupported memory snapshot contract")
    tables = snapshot.get("tables")
    if not isinstance(tables, dict) or set(tables) != set(TABLE_COLUMNS):
        raise ValueError("Memory snapshot table set is invalid")
    for table, columns in TABLE_COLUMNS.items():
        rows = tables[table]
        if not isinstance(rows, list) or any(set(row) != set(columns) for row in rows):
            raise ValueError(f"Memory snapshot columns are invalid: {table}")
        identities = {tuple(row[column] for column in TABLE_KEYS[table]) for row in rows}
        if len(identities) != len(rows):
            raise ValueError(f"Memory snapshot contains duplicate identities: {table}")
    conversations = {(row["workspace_id"], row["principal_id"], row["conversation_id"])
                     for row in tables["conversations"]}
    if any((row["workspace_id"], row["principal_id"], row["conversation_id"]) not in conversations
           for table in ("turns", "conversation_summaries", "episodes") for row in tables[table]):
        raise ValueError("Memory snapshot contains an orphaned conversation reference")
    episodes = {(row["workspace_id"], row["principal_id"], row["episode_id"])
                for row in tables["episodes"]}
    if any((row["workspace_id"], row["principal_id"], row["episode_id"]) not in episodes
           for row in tables["episode_reviews"]):
        raise ValueError("Memory snapshot contains an orphaned episode review")
    for row in tables["turns"]:
        if row["status"] not in {"complete", "failed"}:
            raise ValueError("Running turns may not enter a portable snapshot")
        if row["status"] == "complete":
            if row["response_json"] is None:
                raise ValueError("Completed portable turns require a response")
            json.loads(row["response_json"])
    turns = {(row["workspace_id"], row["principal_id"], row["conversation_id"], row["idempotency_key"])
             for row in tables["turns"]}
    if any((row["workspace_id"], row["principal_id"], row["conversation_id"], row["idempotency_key"])
           not in turns for row in tables["episodes"]):
        raise ValueError("Memory snapshot episode does not reference a completed turn")
    for row in tables["episodes"]:
        for column in ("source_ids_json", "tool_context_json", "evidence_json", "result_json"):
            json.loads(row[column])
    if "sha256" in snapshot:
        payload = {"schema_version": snapshot["schema_version"], "kind": snapshot["kind"],
                   "tables": snapshot["tables"]}
        if snapshot["sha256"] != hashlib.sha256(_canonical(payload)).hexdigest():
            raise ValueError("Memory snapshot checksum does not match its content")
    return {"valid": True, "counts": {table: len(rows) for table, rows in tables.items()}}


def export_memory_snapshot(root: Path) -> dict:
    database = _initialize(Path(root))
    connection = sqlite3.connect(database, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        if connection.execute("SELECT COUNT(*) FROM turns WHERE status='running'").fetchone()[0]:
            raise MemoryMigrationBlocked("Memory snapshot requires running turns to finish or fail")
        tables = {}
        for table, columns in TABLE_COLUMNS.items():
            order = ",".join(columns)
            selected = ",".join(columns)
            tables[table] = [dict(row) for row in connection.execute(
                f"SELECT {selected} FROM {table} ORDER BY {order}"
            ).fetchall()]
        connection.rollback()
    finally:
        connection.close()
    payload = {"schema_version": 1, "kind": "docqa_memory_snapshot", "tables": tables}
    validation = validate_memory_snapshot(payload)
    return {**payload, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(_canonical(payload)).hexdigest(), **validation}


def write_memory_snapshot(snapshot: dict, output: Path) -> None:
    validate_memory_snapshot(snapshot)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(snapshot, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def rehearse_memory_migration(root: Path) -> dict:
    root = Path(root)
    before = export_memory_snapshot(root)
    with tempfile.TemporaryDirectory(prefix="docqa-memory-migration-") as folder:
        copy_root = Path(folder)
        source = sqlite3.connect(root / "conversations.sqlite3", timeout=10)
        destination = sqlite3.connect(copy_root / "conversations.sqlite3")
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
        copied = export_memory_snapshot(copy_root)
    after = export_memory_snapshot(root)
    return {"schema_version": 1, "source_sha256": before["sha256"],
            "copy_sha256": copied["sha256"], "source_after_sha256": after["sha256"],
            "counts": before["counts"], "source_mutated": before["sha256"] != after["sha256"],
            "copy_matches_source": before["sha256"] == copied["sha256"],
            "passed": before["sha256"] == copied["sha256"] == after["sha256"]}
