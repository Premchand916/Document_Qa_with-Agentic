def route_intent(state):

    intent = state.get("intent", "document_qa")

    if intent == "data_analysis":
        return "unsupported_intent"

    return "qa_agent"
