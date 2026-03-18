from agents.orchestrator_agent import create_orchestrator
from agents.tools import retrieve_documents_tool, memory_tool


def react_agent(state):

    query = state["query"]
    documents = retrieve_documents_tool(state)

    context = "\n\n".join([doc.page_content for doc in documents])

    llm = create_orchestrator()

    prompt = f"""
You are an intelligent AI agent.

You can:
1. Retrieve document context
2. Use past conversation memory
3. Reason step-by-step

Follow this format:

Thought:
Action:
Observation:
Final Answer:

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    # 🔥 Fix for your previous bug (list vs string)
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, list):
            content = " ".join([str(x) for x in content])
    else:
        content = str(response)

    state["response"] = content

    return state