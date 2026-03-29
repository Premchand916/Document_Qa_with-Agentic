from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # ── Core query fields ──────────────────────────────────────────────────────
    query: str                          # User's raw question
    prompt_query: str                   # Structured task prompt derived from the raw question
    web_search_query: str               # Search-optimized query for the web search backend
    prompt_skill_path: str              # Path to the active prompt skill/profile file
    intent: str                         # Classified intent (e.g. "factual", "analytical")
    response: str                       # Final answer returned to the UI

    # ── Retrieved content ──────────────────────────────────────────────────────
    documents: List[Any]                # LangChain Document objects (from retriever or web search)
    retrieved_docs: List[Any]           # Raw retrieved docs before dedup/rerank (retriever only)
    source: str                         # Primary source string shown in UI
    sources: List[str]                  # All source references (used by react/writer agents)

    # ── Planning & routing ─────────────────────────────────────────────────────
    task: str                           # Planner-derived task description
    document_type: str                  # Detected document type (pdf, excel, pptx, …)
    use_tool: bool                      # If True, route to dataframe_tool; else retriever
    response_plan: Dict[str, Any]       # Strategy dict from response_planner
    draft_answer: str                   # Intermediate draft (react_agent → writer_agent)

    # ── UI / session context ───────────────────────────────────────────────────
    vectorstore: Any                    # FAISS vectorstore instance
    use_web_search: bool                # If True, bypass docs and query DuckDuckGo
    chat_history: List[Dict[str, str]]  # Prior turns [{"role": ..., "content": ...}]
    answer_mode: str                    # "Auto" | "Concise" | "Detailed" | "Bullet Points"
    audience: str                       # "General" | "Technical" | "Executive"
    response_depth: str                 # "Standard" | "Deep" | "Quick"
    uploaded_file_types: List[str]      # File extensions of uploaded docs (e.g. ["pdf", "xlsx"])
    tabular_assets: List[Dict[str, Any]]  # Parsed tabular assets from Excel/CSV files

    # ── Legacy / unused (kept for backward compatibility) ──────────────────────
    thoughts: List[str]                 # Reserved for chain-of-thought traces
    tools_used: List[str]              # Reserved for tool-use audit log
