import re

from agents.orchestrator_agent import invoke_orchestrator

ALLOWED_INTENTS = {
    "document_qa",
    "data_analysis",
    "summarization",
    "document_comparison",
    "risk_review",
    "action_plan",
    "research_synthesis",
}

ANSWER_MODE_GUIDANCE = {
    "Auto": "Adapt the answer form to the question and the evidence.",
    "Executive Brief": "Keep it concise, strategic, and decision-ready with 3-6 bullets.",
    "Analyst Deep Dive": "Provide a detailed, evidence-rich explanation with structured sections.",
    "Comparison Matrix": "Compare items side by side and use a markdown table when helpful.",
    "Action Plan": "Convert findings into prioritized next steps, owners, and risks when supported.",
    "Risk Review": "Emphasize risks, impact, severity, and mitigation ideas.",
    "Data Highlights": "Focus on metrics, patterns, anomalies, and numerical evidence.",
    "Research Synthesis": "Synthesize themes, insights, tensions, and open questions across files.",
}

AUDIOENCE_GUIDANCE = {
    "General": "Use clear plain-English explanations.",
    "Leadership": "Focus on business implications, decisions, and tradeoffs.",
    "Analyst": "Focus on evidence, assumptions, and data quality.",
    "Product": "Focus on user problems, priorities, opportunities, and requirements.",
    "Operations": "Focus on workflows, blockers, risks, and process improvements.",
    "Sales": "Focus on objections, positioning, customer value, and proof points.",
}

DEPTH_GUIDANCE = {
    "Fast": "Be crisp and short.",
    "Balanced": "Balance brevity with enough detail to be useful.",
    "Comprehensive": "Be thorough and structured, including caveats and evidence.",
}


def _heuristic_intent(query, answer_mode):
    query_lower = query.lower()

    if answer_mode == "Risk Review" or any(word in query_lower for word in ("risk", "compliance", "issue", "concern")):
        return "risk_review"

    if answer_mode == "Action Plan" or any(word in query_lower for word in ("next step", "action", "plan", "roadmap")):
        return "action_plan"

    if answer_mode == "Comparison Matrix" or any(
        word in query_lower for word in ("compare", "difference", "versus", "vs", "contrast")
    ):
        return "document_comparison"

    if answer_mode == "Data Highlights" or any(
        word in query_lower for word in ("metric", "trend", "revenue", "average", "count", "csv", "excel", "sheet")
    ):
        return "data_analysis"

    if answer_mode == "Research Synthesis" or any(
        word in query_lower for word in ("theme", "insight", "synthesis", "patterns", "research")
    ):
        return "research_synthesis"

    if any(word in query_lower for word in ("summarize", "summarise", "summary", "overview")):
        return "summarization"

    return "document_qa"


def _default_sections(intent, answer_mode):
    if answer_mode == "Executive Brief":
        return ["Bottom line", "Key findings", "Implications", "Recommended next step"]
    if answer_mode == "Comparison Matrix":
        return ["Comparison overview", "Key differences", "Shared themes", "Recommendation"]
    if answer_mode == "Action Plan":
        return ["Situation", "Recommended actions", "Priority", "Open risks"]
    if answer_mode == "Risk Review":
        return ["Top risks", "Evidence", "Impact", "Mitigation ideas"]
    if answer_mode == "Data Highlights":
        return ["Top metrics", "Patterns", "Anomalies", "Business meaning"]
    if answer_mode == "Research Synthesis":
        return ["Themes", "Evidence", "Tensions", "Opportunities"]

    if intent == "document_comparison":
        return ["Comparison overview", "Key differences", "Common ground", "Takeaway"]
    if intent == "data_analysis":
        return ["Key numbers", "Patterns", "Anomalies", "Takeaway"]
    if intent == "risk_review":
        return ["Risks", "Evidence", "Impact", "Mitigation"]
    if intent == "action_plan":
        return ["Current state", "Actions", "Priority", "Dependencies"]
    if intent == "research_synthesis":
        return ["Themes", "Insights", "Open questions", "Takeaway"]
    if intent == "summarization":
        return ["Summary", "Key points", "Important details", "Takeaway"]

    return ["Answer", "Evidence", "Nuances", "Takeaway"]


def _parse_line_value(text, label):
    match = re.search(rf"{label}:\s*(.+)", text, re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def plan_response_strategy(state):
    query = state["query"]
    answer_mode = state.get("answer_mode", "Auto")
    audience = state.get("audience", "General")
    response_depth = state.get("response_depth", "Balanced")
    file_types = ", ".join(state.get("uploaded_file_types", [])) or "unknown"

    heuristic_intent = _heuristic_intent(query, answer_mode)

    prompt = f"""
You are a response planning agent for a document intelligence product.
Design the best answer strategy for the user's question before retrieval and generation.

Question: {query}
Selected answer mode: {answer_mode}
Audience: {audience}
Depth: {response_depth}
Uploaded file types: {file_types}

Return exactly this format:
intent: <one of {", ".join(sorted(ALLOWED_INTENTS))}>
user_need: <short phrase>
answer_mode: <best final mode>
output_format: <short phrase>
sections: <section 1 | section 2 | section 3 | section 4>
retrieval_focus: <short phrase>
    """

    try:
        response = invoke_orchestrator(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = " ".join(str(item) for item in content)
        content = str(content)

        intent = _parse_line_value(content, "intent").lower()
        user_need = _parse_line_value(content, "user_need")
        planned_mode = _parse_line_value(content, "answer_mode") or answer_mode
        output_format = _parse_line_value(content, "output_format") or planned_mode
        sections_text = _parse_line_value(content, "sections")
        retrieval_focus = _parse_line_value(content, "retrieval_focus")
    except Exception:
        intent = heuristic_intent
        user_need = "Answer the user's question with evidence from the uploaded files."
        planned_mode = answer_mode
        output_format = answer_mode
        sections_text = ""
        retrieval_focus = "Prioritize the most relevant evidence and cross-document patterns."

    if intent not in ALLOWED_INTENTS:
        intent = heuristic_intent

    if not planned_mode or planned_mode == "Auto":
        planned_mode = answer_mode if answer_mode != "Auto" else "Concise evidence-based answer"

    sections = [section.strip() for section in sections_text.split("|") if section.strip()]
    if not sections:
        sections = _default_sections(intent, answer_mode)

    state["intent"] = intent
    state["response_plan"] = {
        "intent": intent,
        "user_need": user_need or "Answer the user's question using the uploaded evidence.",
        "answer_mode": planned_mode,
        "output_format": output_format or planned_mode,
        "sections": sections[:5],
        "retrieval_focus": retrieval_focus or "Use the strongest evidence from the retrieved documents.",
        "audience_guidance": AUDIOENCE_GUIDANCE.get(audience, AUDIOENCE_GUIDANCE["General"]),
        "depth_guidance": DEPTH_GUIDANCE.get(response_depth, DEPTH_GUIDANCE["Balanced"]),
        "mode_guidance": ANSWER_MODE_GUIDANCE.get(answer_mode, ANSWER_MODE_GUIDANCE["Auto"]),
    }

    return state
