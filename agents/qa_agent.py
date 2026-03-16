from agents.orchestrator_agent import create_orchestrator


from agents.orchestrator_agent import create_orchestrator


def qa_agent(state):
    history = state.get("chat_history", [])

    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in history
    )

    query = state["query"]
    docs = state.get("documents", [])

    if not docs:
        state["draft_answer"] = "No relevant information found."
        return state

    # Build context
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # Collect sources
    sources = []
    for doc in docs:
        meta = doc.metadata
        source = meta.get("source", "Unknown document")
        page = meta.get("page", "N/A")

        sources.append(f"{source} (page {page})")

    prompt = f"""
    Conversation History:
    {history_text}

    Context:
    {context}

    Question:
    {query}

    Answer using the document context and conversation history.
    """

    llm = create_orchestrator()
    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        extracted_text = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                extracted_text.append(item["text"])
            else:
                extracted_text.append(str(item))

        content = " ".join(extracted_text)

    state["draft_answer"] = str(content).strip()

    # Attach citations
    state["sources"] = list(set(sources))

    return state