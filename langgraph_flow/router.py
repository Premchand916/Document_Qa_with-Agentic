def route_intent(state):

    intent = state["intent"]

    if intent == "document_qa":
        return "qa_agent"

    elif intent == "data_analysis":
        return "analysis_agent"

    elif intent == "summarization":
        return "summarizer_agent"

    elif intent == "document_comparison":
        return "comparison_agent"