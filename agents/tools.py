def retrieve_documents_tool(state):
    return state.get("documents", [])


def memory_tool(state):
    return state.get("chat_history", [])