from langchain_core.documents import Document


def web_search_agent(state):
    query = state.get("query", "").strip()
    if not query:
        state["documents"] = []
        state["response"] = "No query provided for web search."
        state["source"] = "Web"
        return state

    try:
        from duckduckgo_search import DDGS

        raw_results = []
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=7))

    except Exception as exc:
        state["documents"] = []
        state["response"] = (
            f"Web search failed: {exc}. "
            "Check your internet connection or try again in a few seconds "
            "(DuckDuckGo rate-limits heavy use)."
        )
        state["source"] = "Web"
        return state

    # ── Build Document objects, skipping results with no body text ────────────
    documents = []
    for result in raw_results:
        url = result.get("href") or result.get("url", "")
        title = result.get("title") or url
        body = (result.get("body") or "").strip()

        # Skip results that have no usable content
        if not body:
            continue

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

    if not documents:
        state["documents"] = []
        state["response"] = (
            "Web search returned no usable results for this query. "
            "Try rephrasing your question or using more specific keywords."
        )
        state["source"] = "Web"
        return state

    state["documents"] = documents
    state["source"] = documents[0].metadata["source"]
    return state
