"""FAISS + BM25 hybrid retrieval with Reciprocal Rank Fusion (RRF)."""
from functools import lru_cache

from rank_bm25 import BM25Okapi

from retrieval.reranker import rerank_documents

_VECTORSTORE_REGISTRY: dict = {}
_BM25_CACHE: dict = {}
RRF_K = 60

# Intent-driven retrieval budget
_INTENT_CONFIG = {
    "data_analysis":       {"k_vector": 4, "k_bm25": 6, "k_final": 4},
    "document_comparison": {"k_vector": 6, "k_bm25": 6, "k_final": 6},
    "summarization":       {"k_vector": 8, "k_bm25": 6, "k_final": 6},
    "research_synthesis":  {"k_vector": 8, "k_bm25": 6, "k_final": 6},
    "risk_review":         {"k_vector": 5, "k_bm25": 5, "k_final": 5},
    "action_plan":         {"k_vector": 5, "k_bm25": 5, "k_final": 4},
    "document_qa":         {"k_vector": 5, "k_bm25": 5, "k_final": 4},
}
_DEFAULT_CONFIG = {"k_vector": 5, "k_bm25": 5, "k_final": 4}


def clear_retrieval_caches() -> None:
    _VECTORSTORE_REGISTRY.clear()
    _BM25_CACHE.clear()
    _cached_vector_search.cache_clear()


@lru_cache(maxsize=200)
def _cached_vector_search(query: str, vectorstore_id: int, k: int):
    vs = _VECTORSTORE_REGISTRY[vectorstore_id]
    return tuple(vs.similarity_search(query, k=k))


def _get_or_build_bm25(vectorstore_id: int, all_docs: tuple):
    if vectorstore_id not in _BM25_CACHE:
        corpus = [doc.page_content.split() for doc in all_docs]
        _BM25_CACHE[vectorstore_id] = (BM25Okapi(corpus), all_docs)
    return _BM25_CACHE[vectorstore_id]


def _rrf_fusion(ranked_lists: list, k: int = RRF_K) -> list:
    """Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank_i + 1) across N lists."""
    scores: dict = {}
    doc_map: dict = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = hash(doc.page_content)
            if key not in scores:
                scores[key] = 0.0
                doc_map[key] = doc
            scores[key] += 1.0 / (k + rank + 1)
    return [doc_map[dk] for dk in sorted(scores, key=lambda dk: scores[dk], reverse=True)]


def retriever_agent(state: dict) -> dict:
    query = state["query"]
    vectorstore = state["vectorstore"]
    intent = state.get("intent", "document_qa")
    cfg = _INTENT_CONFIG.get(intent, _DEFAULT_CONFIG)
    k_vector, k_bm25, k_final = cfg["k_vector"], cfg["k_bm25"], cfg["k_final"]

    vs_id = id(vectorstore)
    _VECTORSTORE_REGISTRY[vs_id] = vectorstore

    # ── FAISS semantic retrieval ──────────────────────────────────────────────
    vector_docs = list(_cached_vector_search(query, vs_id, k_vector))

    # ── BM25 lexical retrieval ────────────────────────────────────────────────
    all_docs_list = list(vectorstore.docstore._dict.values())
    if not all_docs_list:
        state["retrieved_docs"] = vector_docs
        state["documents"] = vector_docs[:k_final]
        return state

    bm25, indexed_docs = _get_or_build_bm25(vs_id, tuple(all_docs_list))
    bm25_scores = bm25.get_scores(query.split())
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k_bm25]
    bm25_docs = [indexed_docs[i] for i in top_bm25_idx]

    # ── RRF fusion ────────────────────────────────────────────────────────────
    fused = _rrf_fusion([vector_docs, bm25_docs])

    # ── Cross-encoder reranking on the fused pool ─────────────────────────────
    candidate_pool = fused[: min(len(fused), k_final * 2)]
    final_docs = rerank_documents(query, candidate_pool, top_k=k_final)

    state["retrieved_docs"] = fused
    state["documents"] = final_docs if final_docs else fused[:k_final]
    return state
