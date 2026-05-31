from pathlib import Path
from functools import lru_cache

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


@lru_cache(maxsize=2)
def get_cached_embedding():
    return get_embedding_model()


def create_vector_store(documents):
    FAISS = _get_faiss_class()

    embedding_model = get_cached_embedding()

    vectorstore = FAISS.from_documents(
        documents,
        embedding_model
    )

    # Ensure DB folder exists
    DB_PATH.mkdir(parents=True, exist_ok=True)

    # Save index
    vectorstore.save_local(str(DB_PATH))

    return vectorstore


def load_vector_store_direct():
    """Load persisted FAISS index without any framework-specific caching."""
    if not DB_PATH.exists():
        return None

    FAISS = _get_faiss_class()
    embedding_model = get_cached_embedding()

    return FAISS.load_local(
        str(DB_PATH),
        embedding_model,
        allow_dangerous_deserialization=True,
    )


# Keep legacy name for backward compatibility with the old Streamlit entry point.
def load_vector_store():
    return load_vector_store_direct()
