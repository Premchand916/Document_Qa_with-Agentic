from pathlib import Path

from embeddings.Hugging_face_embedding import get_embedding_model

DB_PATH = Path(__file__).resolve().parents[1] / "vector_db"


def _get_faiss_class():
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'langchain-community'. Install it with "
            "`pip install -r requirements.txt`."
        ) from exc

    return FAISS


def get_cached_embedding():
    return get_embedding_model()


def build_vector_store(documents):
    """Build in memory; publication is owned by the workspace repository."""
    return _get_faiss_class().from_documents(documents, get_cached_embedding())


def create_vector_store(documents):
    """Compatibility for the legacy entry point, in its own namespace."""
    from storage.workspaces import WorkspaceRepository
    repository = WorkspaceRepository(DB_PATH, get_cached_embedding)
    previous = repository.state("legacy-default")["active_version"]
    vectorstore = build_vector_store(documents)
    repository.publish("legacy-default", vectorstore, [], [], [], [], expected_version=previous)
    return vectorstore


def load_vector_store_direct():
    from storage.workspaces import LegacyIndexError, WorkspaceRepository
    if not DB_PATH.exists():
        return None
    snapshot = WorkspaceRepository(DB_PATH, get_cached_embedding).snapshot("legacy-default")
    if snapshot.vectorstore is None and (DB_PATH / "index.pkl").exists():
        raise LegacyIndexError("Re-upload original files; legacy pickle indexes are not loaded")
    return snapshot.vectorstore


def load_vector_store():
    return load_vector_store_direct()
