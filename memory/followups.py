"""Conservative search-query expansion for explicit follow-up references."""

import re


_REFERENCE = re.compile(r"\b(it|its|they|their|them|this|that|these|those)\b", re.IGNORECASE)
_UNRESOLVED_REFERENCE = re.compile(
    r"\b(?:its|their)\s+(?:deadline|date|status|owner|requirements?|policy|summary)\b|"
    r"\bwhat\s+(?:is|are|about)\s+(?:it|them|this|that)\b",
    re.IGNORECASE,
)


def needs_reference_clarification(question: str, history: list[dict]) -> bool:
    """Detect a narrow set of references that cannot be resolved without prior user context."""
    previous_user = any(message.get("role") == "user" and message.get("content")
                        for message in history or [])
    return not previous_user and bool(_UNRESOLVED_REFERENCE.search(question))


def resolve_followup_query(question: str, history: list[dict]) -> str:
    """Add the prior user question to retrieval, without treating history as evidence."""
    if not _REFERENCE.search(question):
        return question
    previous = next((message["content"] for message in reversed(history)
                     if message.get("role") == "user" and message.get("content")), None)
    return f"{previous}\nFollow-up: {question}" if previous else question
