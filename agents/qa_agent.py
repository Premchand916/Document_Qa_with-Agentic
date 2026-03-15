from agents.orchestrator_agent import create_orchestrator
from tools.vector_search_tool import search_documents


def run_document_qa(state):

    query = state["query"]
    docs = state.get("documents", [])

    if not docs:
        state["response"] = "No relevant information found in the documents."
        return state

    context = "\n\n".join(
        doc.page_content if hasattr(doc, "page_content") else str(doc)
        for doc in docs
    )

    prompt = f"""
    You are a document analysis assistant.

    Use ONLY the provided context to answer the question.

    Context:
    {context}

    Question:
    {query}
    """

    llm = create_orchestrator()

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list): 
        content = " ".join(str(x) for x in content)

    state["response"] = str(content).strip()

    return state

def run_unsupported_intent(state):
    state["response"] = (
        "This app currently supports document-based questions, summaries, and comparisons. "
        "Data-analysis requests are not wired in yet."
    )
    state["thoughts"] = ["The detected intent is not connected to an execution node yet."]
    state["tools_used"] = []
    return state
