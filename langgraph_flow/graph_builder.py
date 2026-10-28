from langgraph.graph import END, StateGraph

from agents.dataframe_tool import dataframe_tool_agent
from agents.evidence import finalize_answer
from agents.planner_agent import planner_agent
from agents.qa_agent import qa_agent
from agents.response_planner import plan_response_strategy
from agents.retriever_agent import retriever_agent
from agents.web_search_agent import web_search_agent
from langgraph_flow.state import AgentState


def route_entry(state):
    return "web_search" if state.get("use_web_search") else "intent_router"


def route_after_planner(state):
    if state.get("use_tool"):
        return "dataframe_tool"
    return "retriever" if state.get("vectorstore") is not None else "knowledge_unavailable"


def knowledge_unavailable(state):
    state["documents"] = []
    state["response"] = "No searchable document text is available for this question. Upload a text-readable document or ask about an available table."
    state["result_status"] = "insufficient_evidence"
    return state


def build_graph():
    workflow = StateGraph(AgentState)

    # Entry passthrough node — routes to web search or document pipeline
    workflow.add_node("entry", lambda s: s)
    workflow.add_node("web_search", web_search_agent)

    # Document pipeline nodes
    workflow.add_node("response_planner", plan_response_strategy)
    workflow.add_node("planner", planner_agent)
    workflow.add_node("dataframe_tool", dataframe_tool_agent)
    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("knowledge_unavailable", knowledge_unavailable)
    workflow.add_node("finalize", finalize_answer)

    # Entry routing
    workflow.set_entry_point("entry")
    workflow.add_conditional_edges(
        "entry",
        route_entry,
        {
            "web_search": "web_search",
            "intent_router": "response_planner",
        },
    )

    # Web search → qa_agent → validated evidence result
    workflow.add_edge("web_search", "qa_agent")

    # Document pipeline
    workflow.add_edge("response_planner", "planner")
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "dataframe_tool": "dataframe_tool",
            "retriever": "retriever",
            "knowledge_unavailable": "knowledge_unavailable",
        },
    )
    workflow.add_edge("dataframe_tool", "finalize")
    workflow.add_edge("retriever", "qa_agent")
    workflow.add_edge("knowledge_unavailable", "finalize")
    workflow.add_edge("qa_agent", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()
