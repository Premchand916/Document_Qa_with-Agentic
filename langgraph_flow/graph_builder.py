from agents.intent_classifier import classify_intent
from agents.qa_agent import run_document_qa, run_unsupported_intent
from langgraph.graph import END, StateGraph
from langgraph_flow.router import route_intent
from langgraph_flow.state import AgentState


def build_graph():

    workflow = StateGraph(AgentState)

    workflow.add_node("intent_classifier", classify_intent)
    workflow.add_node("qa_agent", run_document_qa)
    workflow.add_node("unsupported_intent", run_unsupported_intent)

    workflow.set_entry_point("intent_classifier")

    workflow.add_conditional_edges(
        "intent_classifier",
        route_intent,
        {
            "qa_agent": "qa_agent",
            "unsupported_intent": "unsupported_intent"
        }
    )
    workflow.add_edge("qa_agent", END)
    workflow.add_edge("unsupported_intent", END)

    return workflow.compile()
