from langgraph.graph import StateGraph, END
from langgraph_flow.state import AgentState
from agents.intent_classifier import classify_intent
from langgraph_flow.router import route_intent

def build_graph():

    workflow = StateGraph(AgentState)

    workflow.add_node("intent_classifier", classify_intent)

    workflow.set_entry_point("intent_classifier")

    workflow.add_conditional_edges(
        "intent_classifier",
        route_intent
    )

    return workflow.compile()