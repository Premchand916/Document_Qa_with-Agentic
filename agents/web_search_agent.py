import os

from langchain_core.documents import Document


def _build_search_error(message):
    return (
        f"Web search failed: {message}. "
        "Check your `TAVILY_API_KEY`, internet connection, and Tavily account usage, then try again."
    )


def web_search_agent(state):
    query = state.get("query", "").strip()
    search_query = state.get("web_search_query", query).strip() or query

    if not query:
        state["documents"] = []
        state["response"] = "No query provided for web search."
        state["source"] = "Web"
        return state

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        state["documents"] = []
        state["response"] = (
            "Web search is enabled but `TAVILY_API_KEY` is not configured. "
            "Add it to your `.env` file to use live web search."
        )
        state["source"] = "Web"
        return state

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        raw_response = client.search(
            search_query,
            search_depth="advanced",
            include_answer=True,
            max_results=7,
        )
    except Exception as exc:
        state["documents"] = []
        state["response"] = _build_search_error(str(exc))
        state["source"] = "Web"
        return state

    documents = []
    answer_text = str(raw_response.get("answer") or "").strip()
    if answer_text:
        documents.append(
            Document(
                page_content=f"Tavily answer\n\n{answer_text}",
                metadata={
                    "source": "Tavily",
                    "title": "Tavily answer",
                    "page": "Web",
                    "file_type": "web",
                    "content_type": "web_result",
                    "url": "https://tavily.com",
                },
            )
        )

    for result in raw_response.get("results", []):
        url = str(result.get("url") or "").strip()
        title = str(result.get("title") or url or "Web result").strip()
        body = str(result.get("content") or "").strip()

        if not body:
            continue

        documents.append(
            Document(
                page_content=f"{title}\n\n{body}",
                metadata={
                    "source": url or title,
                    "title": title,
                    "page": "Web",
                    "file_type": "web",
                    "content_type": "web_result",
                    "url": url,
                },
            )
        )

    if not documents:
        state["documents"] = []
        state["response"] = (
            "Tavily returned no usable web results for this query. "
            "Try rephrasing your question or using more specific keywords."
        )
        state["source"] = "Web"
        return state

    state["documents"] = documents
    state["source"] = "Web"
    return state
