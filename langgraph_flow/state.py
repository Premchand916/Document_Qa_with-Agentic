from typing import TypedDict, List

class AgentState(TypedDict):
    query: str
    intent: str
    documents: List[str]
    response: str