from langchain_core.documents import Document


def web_search_agent(state):
    query = state.get("query", "")
    if not query:
        state["documents"] = []
        state["retrieved_docs"] = []
        state["response"] = "No query provided for web search."
        state["source"] = "Web"
        return state

    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as exc:
        state["documents"] = []
        state["retrieved_docs"] = []
        state["response"] = f"Web search failed: {exc}"
        state["source"] = "Web"
        return state

    documents = []
    for result in results:
        url = result.get("href") or result.get("url", "")
        title = result.get("title", url)
        body = result.get("body", "")
        doc = Document(
            page_content=f"{title}\n\n{body}",
            metadata={
                "source": url,
                "title": title,
                "page": "Web",
                "file_type": "web",
                "content_type": "web_result",
                "url": url,
            },
        )
        documents.append(doc)

    state["documents"] = documents
    state["retrieved_docs"] = documents
    state["source"] = documents[0].metadata["source"] if documents else "Web"
    return state
