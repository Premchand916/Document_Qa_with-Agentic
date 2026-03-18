def planner_agent(state):
    query = state["query"]
    tabular_assets = state.get("tabular_assets", [])
    uploaded_file_types = [file_type.lower() for file_type in state.get("uploaded_file_types", [])]

    if "summarize" in query.lower():
        state["task"] = "summarization"
    elif "compare" in query.lower():
        state["task"] = "comparison"
    else:
        state["task"] = "document_qa"

    state["document_type"] = "tabular" if tabular_assets else "document"
    state["use_tool"] = bool(
        tabular_assets and (
            state.get("intent") == "data_analysis"
            or any(file_type in {"csv", "xlsx", "xlsm", "tsv"} for file_type in uploaded_file_types)
        )
    )

    return state
