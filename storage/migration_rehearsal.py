"""Copy-only corpus rollback rehearsal. Source storage is never mutated."""
import hashlib
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Callable

from storage.workspaces import WorkspaceRepository


def _tree_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    paths = [root / "workspaces.sqlite3"]
    if (root / "corpora").exists():
        paths.extend(sorted((root / "corpora").rglob("*")))
    for path in paths:
        if not path.exists():
            continue
        relative = path.relative_to(root).as_posix().encode()
        digest.update(relative)
        if path.is_symlink():
            raise ValueError("Storage copy refuses symbolic links")
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def copy_workspace_store(source_root: Path, destination_root: Path) -> None:
    source_root = Path(source_root).resolve()
    destination_root = Path(destination_root).resolve()
    if destination_root == source_root or source_root in destination_root.parents:
        raise ValueError("Migration copy directory must be outside source storage")
    if not (source_root / "workspaces.sqlite3").is_file():
        raise ValueError("Source workspace database does not exist")
    if destination_root.exists() and any(destination_root.iterdir()):
        raise ValueError("Migration copy directory must be empty")
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Storage copy refuses symbolic links")
    destination_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    corpora = source_root / "corpora"
    if corpora.exists():
        shutil.copytree(corpora, destination_root / "corpora")
    source = sqlite3.connect(f"file:{source_root / 'workspaces.sqlite3'}?mode=ro", uri=True)
    destination = sqlite3.connect(destination_root / "workspaces.sqlite3")
    try:
        source.backup(destination)
        destination.commit()
    finally:
        destination.close()
        source.close()


def rehearse_rollback(source_root: Path, workspace: str, embedding_factory: Callable,
                      *, copy_root: Path | None = None, default_provider: str = "ollama") -> dict:
    source_root = Path(source_root).resolve()
    source_repository = WorkspaceRepository(source_root, embedding_factory, default_provider)
    source_before = source_repository.state(workspace)
    source_fingerprint_before = _tree_fingerprint(source_root)
    versions = source_repository.list_versions(workspace)
    active = source_before["active_version"]
    previous = next((item["version"] for item in versions if item["version"] != active), None)
    if not active or previous is None:
        raise ValueError("Rollback rehearsal requires an active version and one previous version")

    temporary = None
    if copy_root is None:
        temporary = tempfile.TemporaryDirectory()
        rehearsal_root = Path(temporary.name)
    else:
        rehearsal_root = Path(copy_root).resolve()
    try:
        copy_workspace_store(source_root, rehearsal_root)
        copied = WorkspaceRepository(rehearsal_root, embedding_factory, default_provider)
        current_snapshot = copied.snapshot(workspace)
        copied.activate_version(workspace, previous, expected_version=active)
        rolled_back = WorkspaceRepository(rehearsal_root, embedding_factory, default_provider).snapshot(workspace)
        WorkspaceRepository(rehearsal_root, embedding_factory, default_provider).activate_version(
            workspace, active, expected_version=previous
        )
        restored = WorkspaceRepository(rehearsal_root, embedding_factory, default_provider).snapshot(workspace)
        source_after = source_repository.state(workspace)
        source_fingerprint_after = _tree_fingerprint(source_root)
        checks = {
            "copy_started_at_current": current_snapshot.version == active,
            "rollback_loaded_previous": rolled_back.version == previous,
            "restore_loaded_current": restored.version == active,
            "source_active_version_unchanged": source_after["active_version"] == active,
            "source_fingerprint_unchanged": source_fingerprint_after == source_fingerprint_before,
        }
        return {
            "scope": "Offline copy-only rollback rehearsal",
            "workspace_id": workspace,
            "source_root": str(source_root),
            "copy_root": str(rehearsal_root) if copy_root is not None else None,
            "current_version": active,
            "previous_version": previous,
            "checks": checks,
            "passed": all(checks.values()),
            "source_mutated": source_fingerprint_after != source_fingerprint_before,
        }
    finally:
        if temporary is not None:
            temporary.cleanup()
