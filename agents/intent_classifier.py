from agents.orchestrator_agent import create_orchestrator

ALLOWED_INTENTS = {
    "document_qa",
    "data_analysis",
    "summarization",
    "document_comparison",
}


def _heuristic_intent(query):
    query_lower = query.lower()

    if any(keyword in query_lower for keyword in ("summarize", "summarise", "summary")):
        return "summarization"

    if any(keyword in query_lower for keyword in ("compare", "comparison", "difference", "versus", "vs")):
        return "document_comparison"

    if any(
        keyword in query_lower
        for keyword in ("excel", "csv", "spreadsheet", "data", "chart", "plot", "sum", "average", "max", "min")
    ):
        return "data_analysis"

    return "document_qa"


def classify_intent(state):
    query = state["query"]
    prompt = (
        "You are an intent router.\n"
        "Choose exactly one label from:\n"
        "document_qa\n"
        "data_analysis\n"
        "summarization\n"
        "document_comparison\n"
        f"Query: {query}\n"
        "Answer with only the label."
    )

    try:
        llm = create_orchestrator()
        response = llm.invoke(prompt)
        content = response.content

        if isinstance(content, list):
            content = content[0]

        intent = str(content).strip().lower()
    except Exception:
        intent = _heuristic_intent(query)

    if intent not in ALLOWED_INTENTS:
        intent = _heuristic_intent(query)

    state["intent"] = intent

    return state
