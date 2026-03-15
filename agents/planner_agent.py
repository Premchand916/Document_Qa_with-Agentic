def planner_agent(state):

    query = state["query"]

    if "summarize" in query.lower():
        state["task"] = "summarization"
    elif "compare" in query.lower():
        state["task"] = "comparison"
    else:
        state["task"] = "document_qa"

    return state