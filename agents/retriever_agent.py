from functools import lru_cache

from rank_bm25 import BM25Okapi

from retrieval.reranker import rerank_documents

_VECTORSTORE_REGISTRY = {}


@lru_cache(maxsize=100)
def cached_search(query, vectorstore_id):
    vectorstore = _VECTORSTORE_REGISTRY[vectorstore_id]
    return tuple(vectorstore.similarity_search(query, k=3))


def retriever_agent(state):
    query = state["query"]
    vectorstore = state["vectorstore"]
    vectorstore_id = id(vectorstore)
    _VECTORSTORE_REGISTRY[vectorstore_id] = vectorstore

    vector_docs = list(cached_search(query, vectorstore_id))

    all_docs = list(vectorstore.docstore._dict.values())
    corpus = [doc.page_content.split() for doc in all_docs]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.split())

    top_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )[:3]

    keyword_docs = [all_docs[index] for index in top_indices]
    combined_docs = {doc.page_content: doc for doc in vector_docs}

    for doc in keyword_docs:
        combined_docs[doc.page_content] = doc

    merged_docs = list(combined_docs.values())

    tracker = state.get("step_tracker")
    if tracker:
        tracker["status"].write("Reranking evidence by relevance...")

    final_docs = rerank_documents(query, merged_docs[:6], top_k=3)

    state["retrieved_docs"] = final_docs
    state["documents"] = final_docs

    return state
