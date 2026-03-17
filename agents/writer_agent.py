def writer_agent(state):

    answer = state.get("draft_answer", "")
    sources = state.get("sources", [])

    if not answer:
        state["response"] = "Unable to generate answer."
        return state

    formatted_sources = "\n".join(
        f"- {src}" for src in sources
    )

    state["response"] = f"""
{answer}

Sources:
{formatted_sources}
"""

    return state