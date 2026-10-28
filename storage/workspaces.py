"""Immutable corpus artifacts with a transactional SQLite publication pointer."""
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

import pandas as pd
from langchain_core.documents import Document
from embeddings.identity import embedding_identity
from ingestion.semantic_chunker import CHUNKER_VERSION


class PublicationConflict(Exception):
    """Another upload published since this upload started."""


class CorpusIntegrityError(Exception):
    """An active corpus cannot be safely reconstructed."""


class LegacyIndexError(Exception):
    """Legacy pickle indexes require re-upload from original documents."""


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            value.update(chunk)
    return value.hexdigest()


def sync_directory(path):
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write_json(path, value):
    data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def workspace_key(workspace):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", workspace):
        raise ValueError("Invalid workspace identifier")
    return hashlib.sha256(workspace.encode()).hexdigest()


@dataclass(frozen=True)
class CorpusSnapshot:
    workspace_id: str
    provider: str
    version: str | None
    vectorstore: object
    tabular_assets: list
    suggested_questions: list
    file_types: list
    sources: list
    warnings: list = field(default_factory=list)
    extraction_reports: list = field(default_factory=list)


class WorkspaceRepository:
    def __init__(self, root: Path, embedding_factory, default_provider: str = "ollama"):
        if default_provider not in ("gemini", "ollama"):
            raise ValueError("Invalid provider")
        self.root = Path(root)
        self.embedding_factory = embedding_factory
        self.default_provider = default_provider

    @contextmanager
    def _connect(self):
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        sync_directory(self.root.parent)
        connection = sqlite3.connect(self.root / "workspaces.sqlite3", timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS workspaces (
                    workspace_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL CHECK(provider IN ('gemini','ollama')),
                    active_version TEXT);
                CREATE TABLE IF NOT EXISTS versions (
                    workspace_id TEXT NOT NULL, version TEXT NOT NULL,
                    manifest_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(workspace_id, version));
            """)
            yield connection
        finally:
            connection.close()

    def state(self, workspace: str) -> dict:
        workspace_key(workspace)
        with self._connect() as connection:
            connection.execute("INSERT OR IGNORE INTO workspaces VALUES (?, ?, NULL)", (workspace, self.default_provider))
            connection.commit()
            row = connection.execute("""SELECT w.*, v.manifest_hash FROM workspaces w
                LEFT JOIN versions v ON w.workspace_id=v.workspace_id AND w.active_version=v.version
                WHERE w.workspace_id=?""", (workspace,)).fetchone()
            return dict(row)

    def set_provider(self, workspace: str, provider: str) -> None:
        if provider not in ("gemini", "ollama"):
            raise ValueError("Invalid provider")
        self.state(workspace)
        with self._connect() as connection:
            connection.execute("UPDATE workspaces SET provider=? WHERE workspace_id=?", (provider, workspace))
            connection.commit()

    def _version_path(self, workspace, version):
        if not re.fullmatch(r"[a-f0-9]{32}", version):
            raise CorpusIntegrityError("Invalid version")
        return self.root / "corpora" / workspace_key(workspace) / version

    def _commit_publication(self, workspace, version, manifest_hash, expected_version):
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT active_version FROM workspaces WHERE workspace_id=?", (workspace,)).fetchone()[0]
            if current == version:
                existing = connection.execute("""SELECT manifest_hash FROM versions
                    WHERE workspace_id=? AND version=?""", (workspace, version)).fetchone()
                if existing is None or existing["manifest_hash"] != manifest_hash:
                    raise CorpusIntegrityError("Published version metadata differs")
                connection.commit()
                return
            if current != expected_version:
                raise PublicationConflict("Corpus changed during upload; retry against current version")
            connection.execute("INSERT INTO versions(workspace_id,version,manifest_hash) VALUES (?,?,?)", (workspace, version, manifest_hash))
            connection.execute("UPDATE workspaces SET active_version=? WHERE workspace_id=?", (version, workspace))
            connection.commit()

    def publish(self, workspace: str, vectorstore, tabular_assets: list,
                questions: list, file_types: list, sources: list, *, expected_version: str | None,
                warnings: list | None = None, extraction_reports: list | None = None,
                version: str | None = None) -> str:
        import faiss
        if type(vectorstore.index).__name__ != "IndexFlatL2" or vectorstore.index.metric_type != faiss.METRIC_L2:
            raise CorpusIntegrityError("Unsupported vector index metric")
        self.state(workspace)
        version = version or uuid4().hex
        folder = self._version_path(workspace, version)
        if folder.exists():
            manifest_path = folder / "manifest.json"
            if manifest_path.is_file():
                manifest_hash = digest(manifest_path)
                self._load({"workspace_id": workspace, "provider": self.default_provider,
                            "active_version": version, "manifest_hash": manifest_hash})
                self._commit_publication(workspace, version, manifest_hash, expected_version)
                return version
            quarantine = folder.parent / f".{version}.incomplete-{uuid4().hex}"
            folder.rename(quarantine)
        folder.mkdir(parents=True, mode=0o700)
        for parent in (self.root, folder.parent.parent, folder.parent):
            sync_directory(parent)
        documents = []
        for index in range(vectorstore.index.ntotal):
            identifier = vectorstore.index_to_docstore_id[index]
            doc = vectorstore.docstore.search(identifier)
            documents.append({"id": identifier, "page_content": doc.page_content, "metadata": doc.metadata})
        if not documents:
            raise ValueError("Cannot publish an empty corpus")
        encoder = embedding_identity(vectorstore.embedding_function)
        if encoder["dimension"] != vectorstore.index.d:
            raise CorpusIntegrityError("Index and encoder dimensions differ")
        write_json(folder / "documents.json", documents)
        tables = [{"source": asset.get("source", ""), "sheet": asset.get("sheet", ""),
                   "file_type": asset.get("file_type", ""),
                   "table": json.loads(asset["dataframe"].to_json(orient="table", index=False, date_format="iso"))}
                  for asset in tabular_assets]
        write_json(folder / "tables.json", tables)
        write_json(folder / "corpus.json", {"suggested_questions": questions, "file_types": file_types,
                                            "sources": sources, "warnings": warnings or [],
                                            "extraction_reports": extraction_reports or []})
        index_path = folder / "index.faiss"
        faiss.write_index(vectorstore.index, str(index_path))
        index_path.chmod(0o600)
        with index_path.open("rb") as stream:
            os.fsync(stream.fileno())
        manifest = {"format": 2, "workspace_id": workspace, "version": version,
                    "vector_count": len(documents), "dimension": vectorstore.index.d,
                    "embedding": encoder, "metric": "IndexFlatL2", "chunker": CHUNKER_VERSION,
                    "table_count": len(tables),
                    "files": {name: digest(folder / name) for name in ("documents.json", "tables.json", "corpus.json", "index.faiss")}}
        write_json(folder / "manifest.json", manifest)
        sync_directory(folder)
        manifest_hash = digest(folder / "manifest.json")
        # Verify exactly what will be read after restart before exposing this version.
        self._load({"workspace_id": workspace, "provider": self.default_provider,
                    "active_version": version, "manifest_hash": manifest_hash})
        # Never delete on exception: a commit may succeed before its caller fails.
        self._commit_publication(workspace, version, manifest_hash, expected_version)
        return version

    def activate_version(self, workspace: str, version: str, *, expected_version: str | None) -> None:
        """Atomically select an already saved, compatible version for rollback."""
        self.state(workspace)
        with self._connect() as connection:
            row = connection.execute("SELECT manifest_hash FROM versions WHERE workspace_id=? AND version=?",
                                     (workspace, version)).fetchone()
        if row is None:
            raise CorpusIntegrityError("Saved corpus version does not exist")
        self._load({"workspace_id": workspace, "provider": self.default_provider,
                    "active_version": version, "manifest_hash": row["manifest_hash"]})
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT active_version FROM workspaces WHERE workspace_id=?", (workspace,)).fetchone()[0]
            if current != expected_version:
                raise PublicationConflict("Corpus changed during rollback; retry against current version")
            connection.execute("UPDATE workspaces SET active_version=? WHERE workspace_id=?", (version, workspace))
            connection.commit()

    def _verified(self, state):
        folder = self._version_path(state["workspace_id"], state["active_version"])
        if digest(folder / "manifest.json") != state["manifest_hash"]:
            raise CorpusIntegrityError("Manifest checksum mismatch")
        manifest = read_json(folder / "manifest.json")
        if (manifest.get("format") != 2 or manifest.get("workspace_id") != state["workspace_id"]
                or manifest.get("version") != state["active_version"]
                or manifest.get("metric") != "IndexFlatL2"
                or set(manifest.get("files", {})) != {"documents.json", "tables.json", "corpus.json", "index.faiss"}):
            raise CorpusIntegrityError("Invalid manifest")
        for name, expected in manifest["files"].items():
            path = folder / name
            if path.is_symlink() or digest(path) != expected:
                raise CorpusIntegrityError("Artifact checksum mismatch")
        return folder, manifest

    def _compatible_embedding(self, manifest):
        embedding = self.embedding_factory()
        current = embedding_identity(embedding)
        if current != manifest.get("embedding") or current["dimension"] != manifest["dimension"]:
            raise CorpusIntegrityError("Embedding model is incompatible with the active corpus")
        return embedding

    def _load(self, state):
        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore
        from langchain_community.vectorstores import FAISS
        try:
            folder, manifest = self._verified(state)
            embedding = self._compatible_embedding(manifest)
            rows = read_json(folder / "documents.json")
            documents = {row["id"]: Document(page_content=row["page_content"], metadata=row["metadata"]) for row in rows}
            if len(documents) != len(rows) or len(rows) != manifest["vector_count"]:
                raise CorpusIntegrityError("Invalid document count")
            tables = read_json(folder / "tables.json")
            assets = [{"source": row["source"], "sheet": row["sheet"], "file_type": row["file_type"],
                       "dataframe": pd.read_json(StringIO(json.dumps(row["table"])), orient="table")} for row in tables]
            if len(assets) != manifest["table_count"]:
                raise CorpusIntegrityError("Invalid table count")
            corpus = read_json(folder / "corpus.json")
            index = faiss.read_index(str(folder / "index.faiss"))
            if (index.ntotal != len(rows) or index.d != manifest["dimension"]
                    or type(index).__name__ != "IndexFlatL2" or index.metric_type != faiss.METRIC_L2):
                raise CorpusIntegrityError("Invalid vector shape")
            vectorstore = FAISS(embedding, index, InMemoryDocstore(documents),
                                {i: row["id"] for i, row in enumerate(rows)})
            return CorpusSnapshot(state["workspace_id"], state["provider"], state["active_version"],
                                  vectorstore, assets, corpus["suggested_questions"], corpus["file_types"],
                                  corpus["sources"], corpus.get("warnings", []), corpus.get("extraction_reports", []))
        except CorpusIntegrityError:
            raise
        except Exception as exc:
            raise CorpusIntegrityError("Corpus unavailable") from exc

    def snapshot(self, workspace: str) -> CorpusSnapshot:
        state = self.state(workspace)
        if not state["active_version"]:
            return CorpusSnapshot(workspace, state["provider"], None, None, [], [], [], [])
        return self._load(state)

    def describe(self, workspace: str) -> dict:
        state = self.state(workspace)
        result = {"workspace_id": workspace, "provider": state["provider"], "corpus_version": state["active_version"],
                  "vectorstore_ready": False, "tabular_ready": False, "suggested_questions": [], "file_types": [], "sources": [], "warnings": [],
                  "extraction_reports": [],
                  "legacy_index_detected": not state["active_version"] and (self.root / "index.pkl").exists()}
        if state["active_version"]:
            try:
                folder, manifest = self._verified(state)
                self._compatible_embedding(manifest)
                result.update(read_json(folder / "corpus.json"))
                result.update(vectorstore_ready=manifest["vector_count"] > 0, tabular_ready=manifest["table_count"] > 0)
            except Exception as exc:
                raise CorpusIntegrityError("Corpus unavailable") from exc
        return result
