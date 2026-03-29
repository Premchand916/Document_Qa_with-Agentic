from agents.orchestrator_agent import (
    LLMConfigurationError,
    OllamaModelMemoryError,
    invoke_orchestrator,
)
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
    prompt_query = state.get("prompt_query", query)
    documents = retrieve_documents_tool(state)
    history = memory_tool(state)[-6:]
    response_plan = state.get("response_plan", {})
    answer_mode = state.get("answer_mode", "Auto")
    audience = state.get("audience", "General")
    response_depth = state.get("response_depth", "Balanced")
    uploaded_file_types = ", ".join(state.get("uploaded_file_types", [])) or "unknown"

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

    evidence_map = "\n".join(
        f"- {doc.metadata.get('source', 'Unknown')} | "
        f"{doc.metadata.get('page', 'N/A')} | "
        f"{doc.metadata.get('file_type', 'unknown')} | "
        f"{doc.metadata.get('content_type', 'document')}"
        for doc in documents
    )
    planned_sections = response_plan.get("sections", ["Answer", "Evidence", "Takeaway"])

    prompt = f"""
You are an AI R&D assistant answering questions about uploaded documents.
Use only the provided document context and recent conversation.
If the answer is not supported by the context, say that clearly.
Do not reveal chain-of-thought, tool traces, or internal reasoning.
Respond with a direct, polished answer that matches the requested answer mode and audience.

Recent conversation:
{history_text or "No prior conversation."}

Prompt-chain planning:
Intent: {state.get("intent", "document_qa")}
User need: {response_plan.get("user_need", "Answer the user's question with evidence.")}
Answer mode: {response_plan.get("answer_mode", answer_mode)}
Output format: {response_plan.get("output_format", answer_mode)}
Sections to cover: {", ".join(planned_sections)}
Retrieval focus: {response_plan.get("retrieval_focus", "Use the most relevant evidence.")}
Audience guidance: {response_plan.get("audience_guidance", "Use clear language.")}
Depth guidance: {response_plan.get("depth_guidance", "Balance brevity and detail.")}
Mode guidance: {response_plan.get("mode_guidance", "Adapt to the user's needs.")}
Uploaded file types: {uploaded_file_types}
Selected audience: {audience}
Selected depth: {response_depth}
Evidence map:
{evidence_map}

Document context:
{context}

Question:
{query}

Structured task prompt:
{prompt_query}

Answer:
"""

    try:
        response = invoke_orchestrator(prompt)
    except (LLMConfigurationError, OllamaModelMemoryError) as exc:
        state["response"] = str(exc)
        return state

    answer = _extract_final_answer(_extract_text(response))

    if not answer:
        answer = "I could not generate a response from the current document context."

    state["response"] = answer

    return state
