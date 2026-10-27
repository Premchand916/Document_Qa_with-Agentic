from agents.table_plan import build_tool_plan, is_table_request


def planner_agent(state):
    query = state["query"]
    tabular_assets = state.get("tabular_assets", [])

    if "summarize" in query.lower():
        state["task"] = "summarization"
    elif "compare" in query.lower():
        state["task"] = "comparison"
    else:
        state["task"] = "document_qa"

    state["use_tool"] = is_table_request(query, tabular_assets)
    state["document_type"] = "tabular" if state["use_tool"] else "document"
    if state["use_tool"]:
        state["tool_plan"] = build_tool_plan(query, tabular_assets).to_dict()

    return state
