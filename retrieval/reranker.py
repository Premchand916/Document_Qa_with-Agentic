from sentence_transformers import CrossEncoder

# Load once globally (fast inference later)
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_documents(query, documents, top_k=3):

    if not documents:
        return []

    pairs = [(query, doc.page_content) for doc in documents]

    scores = reranker_model.predict(pairs)

    scored_docs = list(zip(documents, scores))

    # Sort by relevance score
    ranked_docs = sorted(
        scored_docs,
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked_docs[:top_k]]