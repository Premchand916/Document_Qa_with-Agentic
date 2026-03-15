from agents.orchestrator_agent import create_orchestrator


from agents.orchestrator_agent import create_orchestrator


def qa_agent(state):

    query = state["query"]
    docs = state.get("documents", [])

    if not docs:
        state["draft_answer"] = "No relevant information found."
        return state

    context = "\n\n".join(
        doc.page_content if hasattr(doc, "page_content") else str(doc)
        for doc in docs
    )

    prompt = f"""
    Answer the question using the provided context.

    Context:
    {context}

    Question:
    {query}
    """

    llm = create_orchestrator()

    response = llm.invoke(prompt)

    content = response.content

    # Extract text safely
    if isinstance(content, list):

        extracted_text = []

        for item in content:
            if isinstance(item, dict) and "text" in item:
                extracted_text.append(item["text"])
            else:
                extracted_text.append(str(item))

        content = " ".join(extracted_text)

    state["draft_answer"] = str(content).strip()

    return state