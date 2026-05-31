"""FastAPI entry point — replaces the Streamlit main.py."""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Any, Dict, List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv(ROOT_DIR / ".env")

from agents.orchestrator_agent import get_active_provider
from agents.question_generator import generate_questions
from agents.retriever_agent import clear_retrieval_caches
from app.prompt_library import PROMPT_LIBRARY, get_prompt_categories
from ingestion.file_loader import extract_tabular_assets, load_uploaded_file
from ingestion.semantic_chunker import semantic_chunk_documents
from langgraph_flow.graph_builder import build_graph
from utils.prompt_skill import build_prompted_query
from vector_Store.faiss_store import create_vector_store, load_vector_store_direct

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Nexus Intelligence", version="2.0")


# ── Session state (single-user; extend with per-session dict for multi-user) ─
class _AppState:
    def __init__(self):
        self.vectorstore = None
        self.tabular_assets: list = []
        self.suggested_questions: list = []
        self.provider: str = get_active_provider()

    def reset(self):
        self.vectorstore = None
        self.tabular_assets = []
        self.suggested_questions = []


_state = _AppState()
_graph = None


@app.on_event("startup")
async def _startup():
    global _graph
    try:
        _state.vectorstore = load_vector_store_direct()
    except Exception:
        _state.vectorstore = None
    _graph = build_graph()


# ── Pydantic models ──────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, Any]] = []
    answer_mode: str = "Auto"
    audience: str = "General"
    response_depth: str = "Balanced"
    use_web_search: bool = False
    uploaded_file_types: List[str] = []


class ProviderRequest(BaseModel):
    provider: str


# ── Bridges FastAPI UploadFile → existing synchronous loaders ─────────────────
class _FileAdapter:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content

    def read(self):
        return self._content

    def seek(self, _pos):
        pass


def _doc_to_dict(doc) -> Dict[str, Any]:
    return {"page_content": doc.page_content, "metadata": dict(doc.metadata)}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    adapted = []
    for f in files:
        content = await f.read()
        adapted.append(_FileAdapter(f.filename or "upload", content))

    raw_docs, tabular_assets = [], []
    for adapter in adapted:
        try:
            raw_docs.extend(load_uploaded_file(adapter))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        tabular_assets.extend(extract_tabular_assets(adapter))

    chunked = semantic_chunk_documents(raw_docs)

    vectorstore = None
    if chunked:
        try:
            vectorstore = create_vector_store(chunked)
        except Exception:
            if not tabular_assets:
                raise

    questions = generate_questions(raw_docs)

    _state.reset()
    clear_retrieval_caches()
    _state.vectorstore = vectorstore
    _state.tabular_assets = tabular_assets
    _state.suggested_questions = questions

    return {
        "questions": questions,
        "chunk_count": len(chunked),
        "source_count": len({d.metadata.get("source", "") for d in raw_docs}),
        "file_types": sorted({Path(a.name).suffix.upper().lstrip(".") for a in adapted}),
        "vectorstore_ready": vectorstore is not None,
        "tabular_ready": bool(tabular_assets),
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if _graph is None:
        raise HTTPException(status_code=503, detail="Workflow not ready.")

    can_run = req.use_web_search or _state.vectorstore is not None or bool(_state.tabular_assets)
    if not can_run:
        raise HTTPException(
            status_code=422,
            detail="Upload at least one file or enable web search first.",
        )

    prompt_query = build_prompted_query(
        req.query,
        {
            "vectorstore": _state.vectorstore,
            "chat_history": req.chat_history,
            "answer_mode": req.answer_mode,
            "audience": req.audience,
            "response_depth": req.response_depth,
            "uploaded_file_types": req.uploaded_file_types,
            "tabular_assets": _state.tabular_assets,
            "use_web_search": req.use_web_search,
        },
    )

    graph_state = {
        "query": req.query,
        "prompt_query": prompt_query,
        "web_search_query": req.query,
        "prompt_skill_path": "",
        "vectorstore": _state.vectorstore,
        "chat_history": req.chat_history,
        "answer_mode": req.answer_mode,
        "audience": req.audience,
        "response_depth": req.response_depth,
        "uploaded_file_types": req.uploaded_file_types,
        "tabular_assets": _state.tabular_assets,
        "use_web_search": req.use_web_search,
    }

    try:
        result = _graph.invoke(graph_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "response": result.get("response", "No answer generated."),
        "source": result.get("source", ""),
        "sources": result.get("sources", []),
        "documents": [_doc_to_dict(d) for d in (result.get("documents") or [])],
        "response_plan": result.get("response_plan", {}),
    }


@app.get("/api/state")
async def get_state():
    return {
        "vectorstore_ready": _state.vectorstore is not None,
        "tabular_ready": bool(_state.tabular_assets),
        "provider": _state.provider,
        "suggested_questions": _state.suggested_questions,
    }


@app.post("/api/provider")
async def set_provider(req: ProviderRequest):
    if req.provider not in ("gemini", "ollama"):
        raise HTTPException(status_code=422, detail="provider must be 'gemini' or 'ollama'.")
    _state.provider = req.provider
    os.environ["LLM_PROVIDER"] = req.provider
    return {"provider": req.provider}


@app.get("/api/prompt-library")
async def get_prompt_library():
    return {
        "categories": get_prompt_categories(),
        "library": {
            k: {"description": v["description"], "prompts": v["prompts"]}
            for k, v in PROMPT_LIBRARY.items()
        },
    }


# ── SPA entry point ───────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
