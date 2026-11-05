"""Principal-scoped connector indexes and read-time corpus composition."""

from hashlib import sha256
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace

from storage.connectors import ConnectorConflict, ConnectorRepository, ConnectorRevoked
from storage.workspaces import CorpusSnapshot, WorkspaceRepository, workspace_key
from ingestion.isolated_parser import parse_in_isolated_process
from ingestion.semantic_chunker import semantic_chunk_documents
from vector_Store.faiss_store import build_vector_store


class ConnectorPublicationError(Exception):
    """Connector content could not be safely indexed."""


def _namespace(workspace: str, principal: str, source_id: str) -> str:
    value = sha256(f"{workspace}\0{principal}\0{source_id}".encode()).hexdigest()
    return f"connector-{value[:48]}"


def _index_repository(data_root: Path, embedding_factory) -> WorkspaceRepository:
    return WorkspaceRepository(Path(data_root) / "connector_indexes", embedding_factory)


def publish_connector_source(repository: ConnectorRepository, data_root: Path,
                             embedding_factory, workspace: str, principal: str,
                             source_id: str) -> dict:
    """Index one current snapshot, then expose it only if source state is unchanged."""
    source = repository.get_source(workspace, principal, source_id)
    if source["state"] != "active" or source["grant_state"] != "active":
        raise ConnectorRevoked()
    snapshot_id = source["last_snapshot_id"]
    if snapshot_id is None:
        raise ConnectorConflict()
    if source["published_snapshot_id"] == snapshot_id:
        return source
    files = repository.load_snapshot_files(workspace, principal, snapshot_id)
    namespace = _namespace(workspace, principal, source_id)
    indexes = _index_repository(data_root, embedding_factory)
    try:
        if files:
            versions = {row["version"] for row in indexes.list_versions(namespace)}
            current = indexes.state(namespace)["active_version"]
            if snapshot_id in versions:
                if current != snapshot_id:
                    indexes.activate_version(namespace, snapshot_id, expected_version=current)
            else:
                raw_documents, tables, warnings, reports, sources = [], [], [], [], []
                for relative, content, provenance in files:
                    display = f"{source['display_name']} [{source_id[:8]}]/{relative}"
                    parsed = parse_in_isolated_process(relative, content)
                    file_id = sha256(f"{source_id}\0{relative}\0{provenance['sha256']}".encode()).hexdigest()
                    extra = {"source": display, "source_id": file_id,
                             "connector_source_id": source_id,
                             "connector_snapshot_id": snapshot_id,
                             "connector_file_sha256": provenance["sha256"],
                             "connector_synced_at": source["last_synced_at"]}
                    for document in parsed.documents:
                        document.metadata.update(extra)
                    for asset in parsed.tabular_assets:
                        asset.update(extra)
                        asset["asset_id"] = f"{display}::{asset['sheet']}"
                    raw_documents.extend(parsed.documents)
                    tables.extend(parsed.tabular_assets)
                    warnings.extend(parsed.warnings)
                    reports.append({**parsed.extraction_report, "source": display})
                    sources.append({"name": display, "sha256": provenance["sha256"],
                                    "connector_source_id": source_id,
                                    "connector_snapshot_id": snapshot_id})
                if not raw_documents:
                    raise ConnectorPublicationError()
                embedding = embedding_factory()
                tokenizer = getattr(getattr(embedding, "client", None), "tokenizer", None)
                chunks = semantic_chunk_documents(raw_documents, tokenizer=tokenizer)
                index = build_vector_store(chunks, embedding=embedding)
                indexes.publish(namespace, index, tables, [],
                                sorted({Path(name).suffix.lower().lstrip(".")
                                        for name, _, _ in files}), sources,
                                expected_version=current, warnings=warnings,
                                extraction_reports=reports, version=snapshot_id)
        return repository.mark_published(workspace, principal, source_id, snapshot_id)
    except (ConnectorConflict, ConnectorRevoked):
        raise
    except Exception as exc:
        raise ConnectorPublicationError() from exc


class CombinedVectorStore:
    """Merge search results while keeping each immutable index separate."""

    def __init__(self, stores: list):
        self.stores = stores
        documents = {}
        for position, store in enumerate(stores):
            for identifier, document in store.docstore._dict.items():
                documents[f"{position}:{identifier}"] = document
        self.docstore = SimpleNamespace(_dict=documents)

    def similarity_search_with_score(self, query: str, k: int = 4):
        results = []
        for store in self.stores:
            results.extend(store.similarity_search_with_score(query, k=k))
        return sorted(results, key=lambda pair: pair[1])[:k]


def composed_snapshot(base: CorpusSnapshot, repository: ConnectorRepository,
                      data_root: Path, embedding_factory, principal: str,
                      authorized_sources: list[dict]) -> tuple[CorpusSnapshot, list[str]]:
    active = [source for source in authorized_sources
              if source["publication_state"] == "ready" and source["freshness"] == "fresh"]
    if not active:
        return base, []
    indexes = _index_repository(data_root, embedding_factory)
    stores = [base.vectorstore] if base.vectorstore is not None else []
    tables, sources, types = list(base.tabular_assets), list(base.sources), set(base.file_types)
    selected = []
    for source in active:
        if source["published_file_count"] == 0:
            selected.append(source["source_id"])
            continue
        namespace = _namespace(base.workspace_id, principal, source["source_id"])
        indexed = indexes.snapshot(namespace)
        if indexed.version != source["published_snapshot_id"]:
            raise ConnectorPublicationError()
        if indexed.vectorstore is not None:
            stores.append(indexed.vectorstore)
        tables.extend(indexed.tabular_assets)
        sources.extend(indexed.sources)
        types.update(indexed.file_types)
        selected.append(source["source_id"])
    repository.assert_published_current(base.workspace_id, principal, selected)
    digest = sha256(repr((base.version, principal,
                          [(source["source_id"], source["published_snapshot_id"])
                           for source in active])).encode()).hexdigest()[:32]
    return CorpusSnapshot(base.workspace_id, base.provider, digest,
                          CombinedVectorStore(stores) if stores else None,
                          tables, base.suggested_questions, sorted(types), sources,
                          base.warnings, base.extraction_reports), selected


def purge_connector_index(data_root: Path, workspace: str, principal: str,
                          source_id: str) -> None:
    """Remove one source's derived index after revocation or deletion."""
    namespace = _namespace(workspace, principal, source_id)
    root = Path(data_root) / "connector_indexes"
    database = root / "workspaces.sqlite3"
    if database.is_file():
        with sqlite3.connect(database) as connection:
            connection.execute("DELETE FROM versions WHERE workspace_id=?", (namespace,))
            connection.execute("DELETE FROM workspaces WHERE workspace_id=?", (namespace,))
            connection.commit()
    folder = root / "corpora" / workspace_key(namespace)
    if folder.is_dir() and not folder.is_symlink():
        shutil.rmtree(folder)
