from rank_bm25 import BM25Okapi


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
    # Keyword Retrieval
    # ---------------------

    all_docs = vectorstore.docstore._dict.values()

    corpus = [doc.page_content.split() for doc in all_docs]

    bm25 = BM25Okapi(corpus)

    tokenized_query = query.split()

    scores = bm25.get_scores(tokenized_query)

    top_n = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:4]

    keyword_docs = [list(all_docs)[i] for i in top_n]

    # ---------------------
    # Merge Results
    # ---------------------

    combined_docs = {doc.page_content: doc for doc in vector_docs}

    for doc in keyword_docs:
        combined_docs[doc.page_content] = doc

    state["documents"] = list(combined_docs.values())

    return state