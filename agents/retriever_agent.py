from rank_bm25 import BM25Okapi
from retrieval.reranker import rerank_documents


def retriever_agent(state):

    query = state["query"]
    vectorstore = state["vectorstore"]

    # ---------------------
    # Vector Retrieval
    # ---------------------
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    vector_docs = vector_retriever.invoke(query)

    # ---------------------
    # Keyword Retrieval (BM25)
    # ---------------------
    all_docs = list(vectorstore.docstore._dict.values())

    corpus = [doc.page_content.split() for doc in all_docs]

    bm25 = BM25Okapi(corpus)

    scores = bm25.get_scores(query.split())

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:4]

    keyword_docs = [all_docs[i] for i in top_indices]

    # ---------------------
    # Merge Results
    # ---------------------
    combined_docs = {doc.page_content: doc for doc in vector_docs}

    for doc in keyword_docs:
        combined_docs[doc.page_content] = doc

    merged_docs = list(combined_docs.values())

    # ---------------------
    # 🔥 Re-ranking (NEW)
    # ---------------------
    final_docs = rerank_documents(query, merged_docs[:6], top_k=2)

    state["documents"] = final_docs

    return state