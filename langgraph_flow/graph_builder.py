from langgraph.graph import StateGraph, END

from agents.planner_agent import planner_agent
from agents.retriever_agent import retriever_agent
from agents.qa_agent import qa_agent
from agents.writer_agent import writer_agent


def build_graph():

    workflow = StateGraph(dict)

    workflow.add_node("planner", planner_agent)
    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("writer", writer_agent)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "qa_agent")
    workflow.add_edge("qa_agent", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile()