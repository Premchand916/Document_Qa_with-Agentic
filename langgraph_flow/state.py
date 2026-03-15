from typing import Any, List, TypedDict


class AgentState(TypedDict, total=False):
    query: str
    intent: str
    documents: List[str]
    response: str
    thoughts: List[str]
    tools_used: List[str]
    vectorstore: Any
