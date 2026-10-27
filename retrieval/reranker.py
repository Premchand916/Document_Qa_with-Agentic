"""Bounded cross-encoder reranking with retryable loading and visible fallback."""

from collections import OrderedDict
from dataclasses import dataclass
import math
import os
from pathlib import Path
from threading import Lock
import time

from sentence_transformers import CrossEncoder

MODEL_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"
LOAD_RETRY_SECONDS = 5.0
MAX_MODELS = 2
MAX_CANDIDATES = 12
MAX_QUERY_CHARS = 1000
MAX_DOCUMENT_CHARS = 3000
_MODEL_STATES: OrderedDict[tuple[str, bool], dict] = OrderedDict()
_MODEL_LOCK = Lock()


@dataclass(frozen=True)
class RerankResult:
    documents: list
    scores: list[float] | None
    model: str | None
    status: str
    degradation_reason: str | None = None

    def to_status(self) -> dict:
        return {"status": self.status, "model": self.model, "scores": self.scores,
                "degradation_reason": self.degradation_reason}


def _model_config() -> tuple[str, bool]:
    name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    raw_local = os.getenv("HF_LOCAL_FILES_ONLY", "true").lower()
    return name, raw_local in {"1", "true", "yes"}


def clear_reranker_cache() -> None:
    with _MODEL_LOCK:
        _MODEL_STATES.clear()


def get_reranker_model():
    """Cache ready models; retry failed loads after bounded backoff."""
    key = _model_config()
    with _MODEL_LOCK:
        now = time.monotonic()
        entry = _MODEL_STATES.get(key)
        if entry:
            _MODEL_STATES.move_to_end(key)
            if entry["model"] is not None:
                return entry["model"]
            if now < entry["retry_at"]:
                return None
        try:
            model = CrossEncoder(key[0], cache_folder=str(MODEL_CACHE_DIR), local_files_only=key[1])
        except Exception:
            _MODEL_STATES[key] = {"model": None, "retry_at": now + LOAD_RETRY_SECONDS}
            model = None
        else:
            _MODEL_STATES[key] = {"model": model, "retry_at": 0.0}
        _MODEL_STATES.move_to_end(key)
        while len(_MODEL_STATES) > MAX_MODELS:
            _MODEL_STATES.popitem(last=False)
        return model


# Existing callers and offline fixtures clear the cache through this attribute.
get_reranker_model.cache_clear = clear_reranker_cache


def rerank_with_status(query: str, documents: list, top_k: int = 3) -> RerankResult:
    if not documents or top_k <= 0:
        return RerankResult([], None, None, "not_needed")
    candidates = list(documents[:MAX_CANDIDATES])
    limit = min(top_k, len(candidates))
    model_name = _model_config()[0]
    try:
        model = get_reranker_model()
    except Exception:
        model = None
    if model is None:
        return RerankResult(candidates[:limit], None, model_name, "fallback", "model_unavailable")
    try:
        pairs = [(query[:MAX_QUERY_CHARS], doc.page_content[:MAX_DOCUMENT_CHARS]) for doc in candidates]
        raw_scores = list(model.predict(pairs))
        if len(raw_scores) != len(candidates):
            raise ValueError("Wrong score count")
        scores = [float(score) for score in raw_scores]
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("Non-finite reranker score")
    except Exception:
        return RerankResult(candidates[:limit], None, model_name, "fallback", "invalid_scores_or_inference")
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:limit]
    return RerankResult([doc for doc, _ in ranked], [score for _, score in ranked], model_name, "ready")


def rerank_documents(query, documents, top_k=3):
    """Compatibility wrapper for callers that only need ranked documents."""
    return rerank_with_status(query, documents, top_k).documents
