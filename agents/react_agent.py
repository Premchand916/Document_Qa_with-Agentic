from agents.orchestrator_agent import create_orchestrator
from agents.tools import memory_tool, retrieve_documents_tool


def _extract_text(response):
    if not hasattr(response, "content"):
        return str(response).strip()

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if hasattr(item, "text") and item.text:
                parts.append(item.text)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return " ".join(parts).strip()

    return str(content).strip()


def _extract_final_answer(text):
    marker = "Final Answer:"
    if marker in text:
        return text.split(marker, 1)[-1].strip()
    return text.strip()


def react_agent(state):
    query = state["query"]
    documents = retrieve_documents_tool(state)
    history = memory_tool(state)[-6:]

    if not documents:
        state["source"] = "No source available"
        state["response"] = (
            "I could not find relevant information in the uploaded documents for that question."
        )
        return state

    context = "\n\n".join(doc.page_content[:1200] for doc in documents)
    history_text = "\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        for message in history
        if message.get("content")
    )

    sources = []
    for doc in documents:
        metadata = doc.metadata
        source = metadata.get("source", "Unknown document")
        page = metadata.get("page", "N/A")
        sources.append(f"{source} (page {page})")

    state["source"] = sources[0]
    state["sources"] = list(dict.fromkeys(sources))

    llm = create_orchestrator()
    prompt = f"""
You are an AI R&D assistant answering questions about uploaded documents.
Use only the provided document context and recent conversation.
If the answer is not supported by the context, say that clearly.
Do not reveal chain-of-thought, tool traces, or internal reasoning.
Respond with a direct, polished answer.

Recent conversation:
{history_text or "No prior conversation."}

Document context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    answer = _extract_final_answer(_extract_text(response))

    if not answer:
        answer = "I could not generate a response from the current document context."

    state["response"] = answer

    return state
