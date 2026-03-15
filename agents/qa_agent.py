from agents.orchestrator_agent import create_orchestrator
from tools.vector_search_tool import search_documents
from tools.retriever_tool import retrieve_documents

def run_document_qa(state):

    query = state["query"]
    vectorstore = state["vectorstore"]

    context = retrieve_documents(query, vectorstore)

    if not context:
        state["response"] = "No relevant information found."
        return state

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
