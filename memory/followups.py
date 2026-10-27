"""Conservative search-query expansion for explicit follow-up references."""

import re


_REFERENCE = re.compile(r"\b(it|its|they|their|them|this|that|these|those)\b", re.IGNORECASE)


def resolve_followup_query(question: str, history: list[dict]) -> str:
    """Add the prior user question to retrieval, without treating history as evidence."""
    if not _REFERENCE.search(question):
        return question
    previous = next((message["content"] for message in reversed(history)
                     if message.get("role") == "user" and message.get("content")), None)
    return f"{previous}\nFollow-up: {question}" if previous else question
