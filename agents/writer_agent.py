def writer_agent(state):

    draft = state.get("draft_answer", "")

    if not draft:
        state["response"] = "Unable to generate answer."
        return state

    state["response"] = draft.strip()

    return state