from langgraph.graph import END, StateGraph

from agents.dataframe_tool import dataframe_tool_agent
from agents.intent_classifier import classify_intent
from agents.planner_agent import planner_agent
from agents.qa_agent import qa_agent
from agents.response_planner import plan_response_strategy
from agents.retriever_agent import retriever_agent


def route_after_planner(state):
    return "dataframe_tool" if state.get("use_tool") else "retriever"


def build_graph():
    workflow = StateGraph(dict)

    workflow.add_node("intent_router", classify_intent)
    workflow.add_node("response_planner", plan_response_strategy)
    workflow.add_node("planner", planner_agent)
    workflow.add_node("dataframe_tool", dataframe_tool_agent)
    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("qa_agent", qa_agent)

    workflow.set_entry_point("intent_router")

    workflow.add_edge("intent_router", "response_planner")
    workflow.add_edge("response_planner", "planner")
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "dataframe_tool": "dataframe_tool",
            "retriever": "retriever",
        }
    )
    workflow.add_edge("dataframe_tool", END)
    workflow.add_edge("retriever", "qa_agent")
    workflow.add_edge("qa_agent", END)

    return workflow.compile()
