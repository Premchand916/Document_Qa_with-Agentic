import os
from functools import lru_cache
from pathlib import Path

from sentence_transformers import CrossEncoder

MODEL_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"


def _env_flag(name):
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    return raw_value.lower() in {"1", "true", "yes"}


def _cached_model_exists(model_name):
    snapshot_dir = MODEL_CACHE_DIR / f"models--{model_name.replace('/', '--')}"
    return snapshot_dir.exists()


def _should_use_local_files_only(model_name):
    configured = _env_flag("HF_LOCAL_FILES_ONLY")
    if configured is not None:
        return configured
    return True


@lru_cache(maxsize=1)
def get_reranker_model():
    model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    try:
        return CrossEncoder(
            model_name,
            cache_folder=str(MODEL_CACHE_DIR),
            local_files_only=_should_use_local_files_only(model_name)
        )
    except Exception:
        return None


def rerank_documents(query, documents, top_k=3):
    if not documents:
        return []

    try:
        reranker_model = get_reranker_model()
        if reranker_model is None:
            return documents[:top_k]
        pairs = [(query, doc.page_content) for doc in documents]
        scores = reranker_model.predict(pairs)
    except Exception:
        return documents[:top_k]

    scored_docs = list(zip(documents, scores))
    ranked_docs = sorted(
        scored_docs,
        key=lambda item: item[1],
        reverse=True
    )

    return [doc for doc, _ in ranked_docs[:top_k]]
