from agents.orchestrator_agent import create_orchestrator


def qa_agent(state):
    query = state["query"]
    docs = state.get("documents", [])

    if docs:
        best_doc = docs[0]
        source = best_doc.metadata.get("source", "Unknown")
        page = best_doc.metadata.get("page", "N/A")
        state["source"] = f"{source} (page {page})"
    else:
        state["source"] = "No source"

    if not docs:
        state["response"] = "No relevant information found."
        return state

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a document QA assistant.

Answer the question ONLY using the provided context.

Context:
{context}

Question:
{query}

Instructions:
- Give clear and structured answer
- Do NOT include reasoning steps
- Do NOT include "Thought", "Action", "Observation"
- Only return final answer
"""

    llm = create_orchestrator()
    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        text = response.content
    else:
        text = str(response)

    if isinstance(text, list):
        extracted_text = []
        for item in text:
            if hasattr(item, "text") and item.text:
                extracted_text.append(item.text)
            elif isinstance(item, dict) and "text" in item:
                extracted_text.append(item["text"])
            else:
                extracted_text.append(str(item))
        text = " ".join(extracted_text)

    state["response"] = str(text).strip()

    return state
