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

def create_vector_store(documents):
    FAISS = _get_faiss_class()

    embedding_model = get_embedding_model()

    vectorstore = FAISS.from_documents(
        documents,
        embedding_model
    )

    # Save index to disk
    vectorstore.save_local(str(DB_PATH))

    return vectorstore


def load_vector_store():
    if not DB_PATH.exists():
        return None

    FAISS = _get_faiss_class()
    embedding_model = get_embedding_model()
    vectorstore = FAISS.load_local(
        str(DB_PATH),
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return vectorstore
