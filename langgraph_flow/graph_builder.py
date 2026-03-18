from langgraph.graph import StateGraph, END

from agents.retriever_agent import retriever_agent
from agents.react_agent import react_agent


def build_graph():

    workflow = StateGraph(dict)

    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("agent", react_agent)

    workflow.set_entry_point("retriever")

    workflow.add_edge("retriever", "agent")
    workflow.add_edge("agent", END)

    return workflow.compile()