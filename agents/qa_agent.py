from agents.orchestrator_agent import (
    LLMConfigurationError,
    OllamaModelMemoryError,
    invoke_orchestrator,
)


def qa_agent(state):
    query = state["query"]
    docs = state.get("documents", [])
    use_web_search = bool(state.get("use_web_search")) or state.get("source") == "Web"

    # ── Detect mode ───────────────────────────────────────────────────────────
    is_web = any(
        d.metadata.get("content_type") == "web_result"
        for d in docs
        if hasattr(d, "metadata")
    ) if docs else use_web_search

    # ── Build source label ────────────────────────────────────────────────────
    if docs:
        best_doc = docs[0]
        source = best_doc.metadata.get("source", "Unknown")
        page = best_doc.metadata.get("page", "N/A")
        state["source"] = f"{source} (page {page})"
    elif not state.get("source"):
        state["source"] = "No source"

    if not docs:
        if state.get("response"):
            return state
        state["response"] = (
            "No relevant information found. "
            + ("Try a different search query." if is_web else
               "Try uploading a document that covers this topic.")
        )
        return state

    # ── Build context from all retrieved docs ─────────────────────────────────
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    # ── Web search prompt (flexible — summarise results) ──────────────────────
    if is_web:
        prompt = f"""You are a helpful web search assistant.

Based on the following web search results, answer the question as completely as possible.
Summarise key information from the results. If the results provide partial information,
share what is available and note any gaps.

Web Search Results:
{context}

Question: {query}

Answer:"""

    # ── Document QA prompt (strict — use only provided context) ───────────────
    else:
        prompt = f"""You are a document QA assistant.

Answer the question using the provided document context below.
Be clear and structured. If the context does not contain enough information,
say what is available and what is missing.

Document Context:
{context}

Question: {query}

Instructions:
- Give a clear, structured answer based on the context
- Do NOT include reasoning steps, "Thought", "Action", or "Observation" markers
- Only return the final answer

Answer:"""

    # ── Call LLM ──────────────────────────────────────────────────────────────
    try:
        response = invoke_orchestrator(prompt)
    except (LLMConfigurationError, OllamaModelMemoryError) as exc:
        state["response"] = str(exc)
        return state

    if hasattr(response, "content"):
        text = response.content
    else:
        text = str(response)

    if isinstance(text, list):
        extracted = []
        for item in text:
            if hasattr(item, "text") and item.text:
                extracted.append(item.text)
            elif isinstance(item, dict) and "text" in item:
                extracted.append(item["text"])
            else:
                extracted.append(str(item))
        text = " ".join(extracted)

    state["response"] = str(text).strip()
    return state
