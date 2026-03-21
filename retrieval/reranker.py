import os
from functools import lru_cache
from pathlib import Path

from sentence_transformers import CrossEncoder

MODEL_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"
LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "true").lower() in {"1", "true", "yes"}


@lru_cache(maxsize=1)
def get_reranker_model():
    model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    return CrossEncoder(
        model_name,
        cache_folder=str(MODEL_CACHE_DIR),
        local_files_only=LOCAL_FILES_ONLY
    )


def rerank_documents(query, documents, top_k=3):
    if not documents:
        return []

    try:
        reranker_model = get_reranker_model()
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

    result = []
    for doc, score in ranked_docs[:top_k]:
        doc.metadata["rerank_score"] = float(score)
        result.append(doc)
    return result
