"""Workspace-scoped FAISS and BM25 retrieval with provenance-preserving fusion."""

from collections import OrderedDict
import re
from threading import Lock

from rank_bm25 import BM25Okapi

from retrieval.reranker import rerank_with_status

RRF_K = 60
TOKENIZER_VERSION = "unicode-identifiers-v2-relevance"
MAX_BM25_INDEXES = 4
_BM25_CACHE: OrderedDict[tuple, tuple] = OrderedDict()
_CACHE_LOCK = Lock()

# Legacy import path. The pinned corpus snapshot now owns vectorstores.
_VECTORSTORE_REGISTRY: dict = {}

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
FALLBACK_MAX_L2_DISTANCE = 1.4
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "the", "this", "to", "what", "when", "where",
    "which", "who", "with",
}


def clear_retrieval_caches() -> None:
    with _CACHE_LOCK:
        _BM25_CACHE.clear()
    _VECTORSTORE_REGISTRY.clear()


def _tokens(text: str) -> list[str]:
    """Case-fold words while preserving complete invoice/version identifiers."""
    return re.findall(r"(?u)\b\w+(?:[-./]\w+)*\b", text.casefold())


def _meaningful_tokens(text: str) -> list[str]:
    return [token for token in _tokens(text)
            if token not in _STOPWORDS and (len(token) > 1 or token.isdigit())]


def _get_or_build_bm25(vectorstore_id: int, all_docs: tuple):
    """Legacy entry point; unscoped callers never retain another corpus."""
    del vectorstore_id
    return BM25Okapi([_tokens(doc.page_content) for doc in all_docs]), all_docs


def _build_bm25_index(all_docs: tuple):
    tokens = [_tokens(doc.page_content) for doc in all_docs]
    if not any(tokens):
        return None, all_docs, [set() for _ in tokens]
    return BM25Okapi(tokens), all_docs, [set(item) for item in tokens]


def _workspace_bm25(state: dict, all_docs: tuple):
    context = state.get("request_context") or {}
    workspace, version = context.get("workspace_id"), state.get("corpus_version")
    if not workspace or not version:
        return _build_bm25_index(all_docs)
    key = (workspace, version, TOKENIZER_VERSION)
    with _CACHE_LOCK:
        cached = _BM25_CACHE.get(key)
        if cached is not None:
            _BM25_CACHE.move_to_end(key)
            return cached
    built = _build_bm25_index(all_docs)
    with _CACHE_LOCK:
        cached = _BM25_CACHE.setdefault(key, built)
        _BM25_CACHE.move_to_end(key)
        while len(_BM25_CACHE) > MAX_BM25_INDEXES:
            _BM25_CACHE.popitem(last=False)
        return cached


def _document_key(doc):
    chunk_id = doc.metadata.get("chunk_id")
    return ("chunk", str(chunk_id)) if chunk_id else ("object", id(doc))


def _rrf_fusion(ranked_lists: list, k: int = RRF_K) -> list:
    """Fuse by stable chunk ID, retaining separate identical-text sources."""
    scores: dict = {}
    doc_map: dict = {}
    for ranked in ranked_lists:
        seen = set()
        for rank, doc in enumerate(ranked):
            key = _document_key(doc)
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map.setdefault(key, doc)
    return [doc_map[key] for key in sorted(scores, key=lambda item: scores[item], reverse=True)]


def retriever_agent(state: dict) -> dict:
    query = state.get("search_query") or state["query"]
    vectorstore = state["vectorstore"]
    cfg = _INTENT_CONFIG.get(state.get("intent"), _DEFAULT_CONFIG)
    k_vector, k_bm25, k_final = cfg["k_vector"], cfg["k_bm25"], cfg["k_final"]

    if hasattr(vectorstore, "similarity_search_with_score"):
        vector_results = list(vectorstore.similarity_search_with_score(query, k=k_vector))
        vector_docs = [document for document, _ in vector_results]
    else:
        vector_docs = vectorstore.similarity_search(query, k=k_vector)
        vector_results = [(document, None) for document in vector_docs]
    all_docs = tuple(vectorstore.docstore._dict.values())
    bm25_docs = []
    if all_docs and _tokens(query):
        bm25, indexed_docs, token_sets = _workspace_bm25(state, all_docs)
        query_tokens = _meaningful_tokens(query)
        query_set = set(query_tokens)
        if bm25 is not None:
            scores = bm25.get_scores(query_tokens)
            minimum_overlap = 1 if len(query_set) < 3 else 2
            matching = [index for index, tokens in enumerate(token_sets)
                        if len(query_set.intersection(tokens)) >= minimum_overlap]
            if matching and max(float(scores[index]) for index in matching) > 0:
                strongest = max(float(scores[index]) for index in matching)
                matching = [index for index in matching
                            if float(scores[index]) >= strongest * 0.5]
            top_indexes = sorted(matching, key=lambda index: scores[index], reverse=True)[:k_bm25]
            bm25_docs = [indexed_docs[index] for index in top_indexes]

    fused = _rrf_fusion([vector_docs, bm25_docs])
    candidates = fused[:min(len(fused), k_final * 2)]
    ranked = rerank_with_status(query, candidates, top_k=k_final)
    documents = ranked.documents
    relevance_gate = {"status": "reranker", "reason": None, "best_l2_distance": None}
    if ranked.status != "ready":
        lexical_keys = {_document_key(document) for document in bm25_docs}
        if lexical_keys:
            documents = [document for document in documents if _document_key(document) in lexical_keys]
            relevance_gate = {"status": "lexical_fallback", "reason": None, "best_l2_distance": None}
        elif vector_results and vector_results[0][1] is not None:
            best_distance = float(vector_results[0][1])
            cross_language = any(ord(character) > 127 for character in query)
            if best_distance <= FALLBACK_MAX_L2_DISTANCE or cross_language:
                documents = [vector_results[0][0]]
                relevance_gate = {"status": "semantic_fallback", "reason": "cross_language" if cross_language else None,
                                  "best_l2_distance": best_distance}
            else:
                documents = []
                relevance_gate = {"status": "insufficient_evidence", "reason": "distance_above_threshold",
                                  "best_l2_distance": best_distance}
    explicit_version = re.search(r"\bversion\s+([A-Za-z0-9_.-]+)\b", query, re.IGNORECASE)
    if explicit_version and documents:
        phrase = f"version {explicit_version.group(1)}".casefold()
        version_matches = [document for document in documents if phrase in
                           f'{document.metadata.get("source", "")} {document.page_content}'.casefold()]
        if not version_matches:
            version_matches = [document for document in candidates if phrase in
                               f'{document.metadata.get("source", "")} {document.page_content}'.casefold()]
        if version_matches:
            documents = version_matches
            relevance_gate = {**relevance_gate, "status": "explicit_version", "reason": phrase}
    state["retrieved_docs"] = fused
    state["documents"] = documents
    state["retrieval_status"] = {
        "vector_hits": len(vector_docs), "lexical_hits": len(bm25_docs),
        "candidate_count": len(candidates), "retrieval_attempts": 1, "max_retrieval_attempts": 1,
        "reranker": ranked.to_status(),
        "relevance_gate": relevance_gate,
    }
    return state
