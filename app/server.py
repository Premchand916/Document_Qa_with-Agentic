"""FastAPI entry point — replaces the Streamlit main.py."""
import asyncio
import hashlib
import os
import sys
from pathlib import Path
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Any, Dict, List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

load_dotenv(ROOT_DIR / ".env")

from agents.orchestrator_agent import get_active_provider, LLMConfigurationError, OllamaModelMemoryError
from agents.question_generator import generate_questions
from app.prompt_library import PROMPT_LIBRARY, get_prompt_categories
from app.schemas import ChatRequest, ProviderRequest
from app.security import BoundaryMiddleware, MAX_FILE_BYTES, MAX_FILES, error_response
from ingestion.file_loader import IngestionLimitError, OCRRequiredError, parse_uploaded_file
from ingestion.isolated_parser import ParserTimeoutError, ParserUnavailableError, parse_in_isolated_process
from ingestion.semantic_chunker import semantic_chunk_documents
from langgraph_flow.graph_builder import build_graph
from memory.followups import resolve_followup_query
from utils.prompt_skill import build_prompted_query
from vector_Store import faiss_store
from vector_Store.faiss_store import build_vector_store
from storage.workspaces import WorkspaceRepository, PublicationConflict, CorpusIntegrityError
from storage.conversations import ConversationRepository, ConversationConflict, ConversationNotFound

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Nexus Intelligence", version="2.0")
app.add_middleware(BoundaryMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    return error_response(422, "validation_error", "Request validation failed. Check required fields, allowed values and size limits.", request.state.request_id)


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    messages = {
        400: ("invalid_request", "The request could not be parsed."),
        413: ("request_too_large", "The upload exceeds the file count or byte limit."),
        415: ("unsupported_media_type", "Upload files using multipart/form-data."),
        422: ("validation_error", "Upload readable supported files and check request values."),
        503: ("service_unavailable", "The workflow or model provider is unavailable."),
    }
    code, detail = messages.get(exc.status_code, ("request_failed", "The request could not be completed."))
    return error_response(exc.status_code, code, detail, request.state.request_id)


def get_repository() -> WorkspaceRepository:
    root = Path(os.getenv("DOCQA_DATA_DIR") or faiss_store.DB_PATH)
    return WorkspaceRepository(root, lambda: faiss_store.get_cached_embedding(), get_active_provider())


def get_conversation_repository() -> ConversationRepository:
    return ConversationRepository(Path(os.getenv("DOCQA_DATA_DIR") or faiss_store.DB_PATH))


@app.exception_handler(PublicationConflict)
async def publication_conflict(request: Request, exc: PublicationConflict):
    return error_response(409, "publication_conflict", "Another upload replaced this workspace. Refresh and retry.", request.state.request_id)


@app.exception_handler(CorpusIntegrityError)
async def corpus_unavailable(request: Request, exc: CorpusIntegrityError):
    return error_response(503, "corpus_unavailable", "Stored corpus failed validation. Re-upload original files.", request.state.request_id)


@app.exception_handler(ConversationNotFound)
async def conversation_not_found(request: Request, exc: ConversationNotFound):
    return error_response(404, "conversation_not_found", "Conversation was not found in this workspace.", request.state.request_id)


@app.exception_handler(ConversationConflict)
async def conversation_conflict(request: Request, exc: ConversationConflict):
    return error_response(409, "conversation_conflict", "Conversation changed or already has a running turn.", request.state.request_id)


_graph = None
_PARSE_SLOTS = asyncio.Semaphore(2)
_CHAT_SLOTS = asyncio.Semaphore(2)


@app.on_event("startup")
async def _startup():
    global _graph
    await asyncio.to_thread(get_conversation_repository().recover_interrupted)
    _graph = build_graph()


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


async def _parse_adapter(adapter):
    async with _PARSE_SLOTS:
        task = asyncio.create_task(asyncio.to_thread(parse_in_isolated_process, adapter.name, adapter.getvalue()))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await task  # Keep the capacity slot until the worker exits.
            except Exception:
                pass
            raise


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_files(request: Request):
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data;"):
        raise HTTPException(status_code=415)
    repository = get_repository()
    workspace = request.state.request_context["workspace_id"]
    previous = repository.state(workspace)
    adapted = []
    try:
        async with request.form(max_files=MAX_FILES, max_fields=0, max_part_size=64 * 1024) as form:
            files = list(form.multi_items())
            if not files or any(key != "files" or not isinstance(value, UploadFile) for key, value in files):
                raise HTTPException(status_code=422)
            for _, file in files:
                name = file.filename or ""
                if not name or len(name) > 255 or any(char in name for char in ("/", "\\", "\x00")):
                    raise HTTPException(status_code=422)
                content = bytearray()
                while chunk := await file.read(64 * 1024):
                    if len(content) + len(chunk) > MAX_FILE_BYTES:
                        raise HTTPException(status_code=413)
                    content.extend(chunk)
                if not content:
                    raise HTTPException(status_code=422)
                adapted.append(_FileAdapter(name, bytes(content)))
    except StarletteHTTPException as exc:
        if exc.status_code == 400 and "Too many files" in str(exc.detail):
            raise HTTPException(status_code=413) from None
        raise

    raw_docs, tabular_assets, warnings, extraction_reports = [], [], [], []
    for adapter in adapted:
        try:
            parsed = await _parse_adapter(adapter)
            raw_docs.extend(parsed.documents)
            tabular_assets.extend(parsed.tabular_assets)
            warnings.extend(parsed.warnings)
            extraction_reports.append(parsed.extraction_report)
        except IngestionLimitError:
            raise HTTPException(status_code=413) from None
        except OCRRequiredError:
            return error_response(422, "ocr_required", "PDF contains no extractable text. Run OCR before uploading it.", request.state.request_id)
        except ParserTimeoutError:
            return error_response(504, "parser_timeout", "Document parsing exceeded its time limit.", request.state.request_id)
        except ParserUnavailableError:
            return error_response(503, "parser_unavailable", "Document parsing is temporarily unavailable.", request.state.request_id)
        except Exception:
            raise HTTPException(status_code=422) from None

    embedding = faiss_store.get_cached_embedding()
    tokenizer = getattr(getattr(embedding, "client", None), "tokenizer", None)
    chunked = semantic_chunk_documents(raw_docs, tokenizer=tokenizer)

    if not chunked:
        raise HTTPException(status_code=422)
    vectorstore = build_vector_store(chunked)
    questions = generate_questions(raw_docs, provider=previous["provider"])
    file_types = sorted({Path(a.name).suffix.lower().lstrip(".") for a in adapted})
    sources = [{"name": a.name, "sha256": hashlib.sha256(a.getvalue()).hexdigest()} for a in adapted]
    version = repository.publish(workspace, vectorstore, tabular_assets, questions, file_types,
                                 sources, expected_version=previous["active_version"], warnings=warnings,
                                 extraction_reports=extraction_reports)

    return {
        "workspace_id": workspace,
        "corpus_version": version,
        "questions": questions,
        "chunk_count": len(chunked),
        "source_count": len({d.metadata.get("source", "") for d in raw_docs}),
        "file_types": file_types,
        "vectorstore_ready": vectorstore is not None,
        "tabular_ready": bool(tabular_assets),
        "warnings": warnings,
        "extraction_reports": extraction_reports,
    }


def _run_chat_graph(req: ChatRequest, context: dict, history: list[dict], conversation_id: str,
                    idempotency_key: str) -> dict:
    snapshot = get_repository().snapshot(context["workspace_id"])
    can_run = req.use_web_search or snapshot.vectorstore is not None or bool(snapshot.tabular_assets)
    if not can_run:
        raise HTTPException(
            status_code=422,
            detail="Upload at least one file or enable web search first.",
        )

    prompt_query = build_prompted_query(
        req.query,
        {
            "vectorstore": snapshot.vectorstore,
            "chat_history": history,
            "answer_mode": req.answer_mode,
            "audience": req.audience,
            "response_depth": req.response_depth,
            "uploaded_file_types": snapshot.file_types,
            "tabular_assets": snapshot.tabular_assets,
            "use_web_search": req.use_web_search,
        },
    )

    graph_state = {
        "query": req.query,
        "search_query": resolve_followup_query(req.query, history),
        "prompt_query": prompt_query,
        "web_search_query": resolve_followup_query(req.query, history),
        "prompt_skill_path": "",
        "vectorstore": snapshot.vectorstore,
        "chat_history": history,
        "request_context": context,
        "provider": snapshot.provider,
        "corpus_version": snapshot.version,
        "answer_mode": req.answer_mode,
        "audience": req.audience,
        "response_depth": req.response_depth,
        "uploaded_file_types": snapshot.file_types,
        "tabular_assets": snapshot.tabular_assets,
        "use_web_search": req.use_web_search,
    }

    try:
        result = _graph.invoke(graph_state)
    except (LLMConfigurationError, OllamaModelMemoryError):
        raise HTTPException(status_code=503) from None

    return {
        "conversation_id": conversation_id,
        "idempotency_key": idempotency_key,
        "workspace_id": snapshot.workspace_id,
        "corpus_version": snapshot.version,
        "response": result.get("response", "No answer generated."),
        "source": result.get("source", ""),
        "sources": result.get("sources", []),
        "documents": [_doc_to_dict(d) for d in (result.get("documents") or [])],
        "response_plan": result.get("response_plan", {}),
        "result_status": result.get("result_status"),
        "tool_result": result.get("tool_result"),
    }


async def _run_bounded_chat(req: ChatRequest, context: dict, history: list[dict],
                            conversation_id: str, idempotency_key: str) -> dict:
    try:
        await asyncio.wait_for(_CHAT_SLOTS.acquire(), timeout=1.0)
    except TimeoutError:
        raise HTTPException(status_code=503, detail="Chat capacity is full. Retry shortly.") from None
    try:
        task = asyncio.create_task(asyncio.to_thread(_run_chat_graph, req, context, history,
                                                      conversation_id, idempotency_key))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await task  # Keep the slot until the blocking model call exits.
            except Exception:
                pass
            raise
    finally:
        _CHAT_SLOTS.release()


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    if _graph is None:
        raise HTTPException(status_code=503, detail="Workflow not ready.")
    context = request.state.request_context
    repository = get_conversation_repository()
    idempotency_key = req.idempotency_key or uuid4().hex
    reserved = await asyncio.to_thread(repository.reserve, context["workspace_id"],
                                       context["principal_id"], req.conversation_id,
                                       idempotency_key, req.query)
    if reserved.cached_response is not None:
        return reserved.cached_response
    try:
        response = await _run_bounded_chat(req, context, reserved.history,
                                           reserved.conversation_id, idempotency_key)
        await asyncio.to_thread(repository.complete, context["workspace_id"], context["principal_id"],
                                reserved.conversation_id, idempotency_key, response)
        return response
    except BaseException:
        await asyncio.to_thread(repository.fail, context["workspace_id"], context["principal_id"],
                                reserved.conversation_id, idempotency_key)
        raise


@app.get("/api/conversations")
async def list_conversations(request: Request):
    context = request.state.request_context
    conversations = await asyncio.to_thread(get_conversation_repository().list_recent,
                                             context["workspace_id"], context["principal_id"])
    return {"conversations": conversations}


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, request: Request):
    context = request.state.request_context
    return await asyncio.to_thread(get_conversation_repository().get,
                                   context["workspace_id"], context["principal_id"], conversation_id)


@app.get("/api/state")
async def get_state(request: Request):
    return get_repository().describe(request.state.request_context["workspace_id"])


@app.get("/api/workspaces")
async def get_workspaces(request: Request):
    context = request.state.request_context
    return {"workspace_id": context["workspace_id"], "workspaces": context["allowed_workspaces"],
            "can_select": context["can_select_workspace"]}


@app.post("/api/provider")
async def set_provider(req: ProviderRequest, request: Request):
    get_repository().set_provider(request.state.request_context["workspace_id"], req.provider)
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
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=True, proxy_headers=False)
