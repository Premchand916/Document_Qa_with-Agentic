"""FastAPI entry point — replaces the Streamlit main.py."""
import asyncio
from contextlib import suppress
import hashlib
import os
import re
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Any, Dict, List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

load_dotenv(ROOT_DIR / ".env")

from agents.orchestrator_agent import get_active_provider, LLMConfigurationError, OllamaModelMemoryError
from agents.question_generator import generate_questions
from app.prompt_library import PROMPT_LIBRARY, get_prompt_categories
from app.schemas import ChatRequest, EpisodeFeedbackRequest, ProviderRequest, SemanticMemoryRequest
from app.security import BoundaryMiddleware, MAX_FILE_BYTES, MAX_FILES, error_response
from ingestion.file_loader import (IngestionLimitError, OCRConfigurationError, OCRRequiredError,
                                   parse_uploaded_file)
from ingestion.isolated_parser import ParserTimeoutError, ParserUnavailableError, parse_in_isolated_process
from ingestion.semantic_chunker import semantic_chunk_documents
from langgraph_flow.graph_builder import build_graph
from memory.followups import needs_reference_clarification, resolve_followup_query
from memory.episodic_memory import EpisodeConflict, EpisodeNotFound, EpisodicMemoryRepository
from memory.semantic_memory import (SemanticMemoryNotFound, SemanticMemoryRejected,
                                    SemanticMemoryRepository)
from utils.prompt_skill import build_prompted_query
from vector_Store import faiss_store
from vector_Store.faiss_store import build_vector_store
from storage.workspaces import WorkspaceRepository, PublicationConflict, CorpusIntegrityError
from storage.conversations import ConversationRepository, ConversationConflict, ConversationNotFound
from storage.ingestion_jobs import (IngestionCancelled, IngestionCapacityError,
                                    IngestionJobConflict, IngestionJobNotFound,
                                    IngestionJobRepository)

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
    def configured_integer(name: str, default: int, minimum: int, maximum: int) -> int:
        raw_value = os.getenv(name, str(default)).strip()
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise RuntimeError(f"{name} must be an integer") from exc
        if not minimum <= value <= maximum:
            raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
        return value

    return ConversationRepository(
        Path(os.getenv("DOCQA_DATA_DIR") or faiss_store.DB_PATH),
        lease_seconds=configured_integer("DOCQA_TURN_LEASE_SECONDS", 60, 10, 3600),
        retention_days=configured_integer("DOCQA_CONVERSATION_RETENTION_DAYS", 90, 1, 3650),
    )


def get_episodic_repository() -> EpisodicMemoryRepository:
    raw_value = os.getenv("DOCQA_EPISODE_RETENTION_DAYS", "180").strip()
    try:
        retention_days = int(raw_value)
    except ValueError as exc:
        raise RuntimeError("DOCQA_EPISODE_RETENTION_DAYS must be an integer") from exc
    return EpisodicMemoryRepository(
        Path(os.getenv("DOCQA_DATA_DIR") or faiss_store.DB_PATH),
        retention_days=retention_days,
    )


def get_semantic_repository() -> SemanticMemoryRepository:
    raw_value = os.getenv("DOCQA_SEMANTIC_RETENTION_DAYS", "365").strip()
    try:
        retention_days = int(raw_value)
    except ValueError as exc:
        raise RuntimeError("DOCQA_SEMANTIC_RETENTION_DAYS must be an integer") from exc
    return SemanticMemoryRepository(
        Path(os.getenv("DOCQA_DATA_DIR") or faiss_store.DB_PATH),
        retention_days=retention_days,
    )


def get_ingestion_repository() -> IngestionJobRepository:
    def configured(name: str, default: int, minimum: int, maximum: int) -> int:
        raw_value = os.getenv(name, str(default)).strip()
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise RuntimeError(f"{name} must be an integer") from exc
        if not minimum <= value <= maximum:
            raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
        return value

    return IngestionJobRepository(
        Path(os.getenv("DOCQA_DATA_DIR") or faiss_store.DB_PATH),
        lease_seconds=configured("DOCQA_INGESTION_LEASE_SECONDS", 120, 10, 3600),
        retention_days=configured("DOCQA_INGESTION_RETENTION_DAYS", 7, 1, 365),
        max_pending_per_workspace=configured("DOCQA_INGESTION_MAX_PENDING", 10, 1, 100),
    )


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


@app.exception_handler(EpisodeNotFound)
async def episode_not_found(request: Request, exc: EpisodeNotFound):
    return error_response(404, "episode_not_found", "Saved feedback was not found.", request.state.request_id)


@app.exception_handler(EpisodeConflict)
async def episode_conflict(request: Request, exc: EpisodeConflict):
    return error_response(409, "episode_conflict", "Feedback already exists for this completed turn.", request.state.request_id)


@app.exception_handler(SemanticMemoryNotFound)
async def semantic_memory_not_found(request: Request, exc: SemanticMemoryNotFound):
    return error_response(404, "semantic_memory_not_found", "Saved memory was not found.", request.state.request_id)


@app.exception_handler(SemanticMemoryRejected)
async def semantic_memory_rejected(request: Request, exc: SemanticMemoryRejected):
    return error_response(422, "semantic_memory_rejected", str(exc), request.state.request_id)


@app.exception_handler(IngestionJobNotFound)
async def ingestion_job_not_found(request: Request, exc: IngestionJobNotFound):
    return error_response(404, "ingestion_job_not_found", "Ingestion job was not found in this workspace.", request.state.request_id)


@app.exception_handler(IngestionJobConflict)
async def ingestion_job_conflict(request: Request, exc: IngestionJobConflict):
    return error_response(409, "ingestion_job_conflict", "Ingestion request conflicts with existing work.", request.state.request_id)


@app.exception_handler(IngestionCapacityError)
async def ingestion_capacity(request: Request, exc: IngestionCapacityError):
    return error_response(503, "ingestion_capacity", "Workspace ingestion capacity is full. Retry shortly.", request.state.request_id)


_graph = None
_CHAT_SLOTS = asyncio.Semaphore(2)
_INGESTION_SLOTS = asyncio.Semaphore(2)
_INGESTION_TASKS: dict[str, asyncio.Task] = {}


@app.on_event("startup")
async def _startup():
    global _graph
    repository = get_conversation_repository()
    await asyncio.to_thread(repository.recover_expired)
    await asyncio.to_thread(repository.purge_expired)
    await asyncio.to_thread(get_episodic_repository().purge_expired)
    await asyncio.to_thread(get_semantic_repository().purge_expired)
    jobs = get_ingestion_repository()
    await asyncio.to_thread(jobs.recover_expired)
    await asyncio.to_thread(jobs.purge_expired)
    _graph = build_graph()
    for job_id in await asyncio.to_thread(jobs.queued_ids):
        _ensure_ingestion_task(job_id)


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


class _EmptyIngestionError(Exception):
    pass


class ChatCapacityError(Exception):
    pass


class ChatDeadlineError(Exception):
    pass


class ProviderUnavailableError(Exception):
    pass


@app.exception_handler(ChatCapacityError)
async def chat_capacity(request: Request, exc: ChatCapacityError):
    return error_response(503, "chat_capacity", "Chat capacity is full. Retry shortly.", request.state.request_id)


@app.exception_handler(ChatDeadlineError)
async def chat_deadline(request: Request, exc: ChatDeadlineError):
    return error_response(504, "chat_deadline", "Chat exceeded its response deadline.", request.state.request_id)


@app.exception_handler(ProviderUnavailableError)
async def provider_unavailable(request: Request, exc: ProviderUnavailableError):
    return error_response(503, "provider_unavailable", "Model provider is unavailable.", request.state.request_id)


def _process_ingestion_job(job_repository: IngestionJobRepository, job_id: str,
                           lease_token: str) -> dict:
    job, inputs = job_repository.load_inputs(job_id, lease_token)
    adapted = [_FileAdapter(name, content) for name, content in inputs]
    repository = get_repository()
    raw_docs, tabular_assets, warnings, extraction_reports = [], [], [], []

    started = time.monotonic()
    job_repository.set_stage(job_id, lease_token, "parsing")
    for adapter in adapted:
        parsed = parse_in_isolated_process(adapter.name, adapter.getvalue())
        raw_docs.extend(parsed.documents)
        tabular_assets.extend(parsed.tabular_assets)
        warnings.extend(parsed.warnings)
        extraction_reports.append(parsed.extraction_report)
    job_repository.record_stage(job_id, lease_token, "parsing", int((time.monotonic() - started) * 1000))
    job_repository.check_cancelled(job_id, lease_token)

    started = time.monotonic()
    job_repository.set_stage(job_id, lease_token, "chunking")
    embedding = faiss_store.get_cached_embedding()
    tokenizer = getattr(getattr(embedding, "client", None), "tokenizer", None)
    chunked = semantic_chunk_documents(raw_docs, tokenizer=tokenizer)
    if not chunked:
        raise _EmptyIngestionError()
    job_repository.record_stage(job_id, lease_token, "chunking", int((time.monotonic() - started) * 1000))
    job_repository.check_cancelled(job_id, lease_token)

    started = time.monotonic()
    job_repository.set_stage(job_id, lease_token, "indexing")
    vectorstore = build_vector_store(chunked)
    job_repository.record_stage(job_id, lease_token, "indexing", int((time.monotonic() - started) * 1000))
    job_repository.check_cancelled(job_id, lease_token)

    started = time.monotonic()
    job_repository.set_stage(job_id, lease_token, "questions")
    provider = repository.state(job["workspace_id"])["provider"]
    questions = generate_questions(raw_docs, provider=provider)
    job_repository.record_stage(job_id, lease_token, "questions", int((time.monotonic() - started) * 1000))
    job_repository.check_cancelled(job_id, lease_token)

    file_types = sorted({Path(a.name).suffix.lower().lstrip(".") for a in adapted})
    sources = [{"name": a.name, "sha256": hashlib.sha256(a.getvalue()).hexdigest()} for a in adapted]
    started = time.monotonic()
    job_repository.set_stage(job_id, lease_token, "publication")
    version = repository.publish(job["workspace_id"], vectorstore, tabular_assets, questions, file_types,
                                 sources, expected_version=job["expected_version"], warnings=warnings,
                                 extraction_reports=extraction_reports, version=job_id)
    job_repository.record_stage(job_id, lease_token, "publication", int((time.monotonic() - started) * 1000))

    return {
        "ingestion_job_id": job_id,
        "workspace_id": job["workspace_id"],
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


def _safe_ingestion_failure(exc: BaseException) -> tuple[str, str, int]:
    if isinstance(exc, IngestionLimitError):
        return "request_too_large", "Document exceeds a configured ingestion limit.", 413
    if isinstance(exc, OCRRequiredError):
        return "ocr_required", "PDF contains no extractable text. Run OCR before uploading it.", 422
    if isinstance(exc, OCRConfigurationError):
        return "ocr_not_configured", "Requested local OCR language packs are unavailable.", 503
    if isinstance(exc, ParserTimeoutError):
        return "parser_timeout", "Document parsing exceeded its time limit.", 504
    if isinstance(exc, ParserUnavailableError):
        return "parser_unavailable", "Document parsing is temporarily unavailable.", 503
    if isinstance(exc, PublicationConflict):
        return "publication_conflict", "Another upload replaced this workspace. Refresh and retry.", 409
    if isinstance(exc, CorpusIntegrityError):
        return "corpus_unavailable", "Corpus publication failed validation.", 503
    if isinstance(exc, (LLMConfigurationError, OllamaModelMemoryError)):
        return "provider_unavailable", "Model provider is unavailable.", 503
    if isinstance(exc, _EmptyIngestionError):
        return "empty_document", "Upload contains no indexable content.", 422
    if isinstance(exc, ValueError):
        return "invalid_document", "Upload is unsupported or could not be parsed.", 422
    return "ingestion_failed", "Ingestion failed before publication.", 500


async def _maintain_ingestion_lease(repository: IngestionJobRepository, job_id: str,
                                    lease_token: str, lease_lost: asyncio.Event) -> None:
    interval = max(1, repository.lease_seconds // 3)
    while True:
        await asyncio.sleep(interval)
        try:
            await asyncio.to_thread(repository.renew, job_id, lease_token)
        except Exception:
            lease_lost.set()
            return


async def _run_ingestion_job(job_id: str) -> None:
    async with _INGESTION_SLOTS:
        repository = get_ingestion_repository()
        lease_token = await asyncio.to_thread(repository.claim, job_id)
        if lease_token is None:
            return
        lease_lost = asyncio.Event()
        heartbeat = asyncio.create_task(_maintain_ingestion_lease(
            repository, job_id, lease_token, lease_lost
        ))
        try:
            result = await asyncio.to_thread(_process_ingestion_job, repository, job_id, lease_token)
            if lease_lost.is_set():
                raise IngestionJobConflict("Ingestion lease was lost")
            await asyncio.to_thread(repository.complete, job_id, lease_token, result)
        except asyncio.CancelledError:
            raise
        except IngestionCancelled:
            await asyncio.to_thread(repository.finish_cancelled, job_id, lease_token)
        except BaseException as exc:
            code, message, status = _safe_ingestion_failure(exc)
            with suppress(IngestionJobConflict):
                await asyncio.to_thread(repository.fail, job_id, lease_token, code, message, status)
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat


def _ensure_ingestion_task(job_id: str) -> asyncio.Task:
    existing = _INGESTION_TASKS.get(job_id)
    if existing is not None and not existing.done():
        return existing
    task = asyncio.create_task(_run_ingestion_job(job_id))
    _INGESTION_TASKS[job_id] = task
    task.add_done_callback(lambda completed, identifier=job_id:
                           _INGESTION_TASKS.pop(identifier, None)
                           if _INGESTION_TASKS.get(identifier) is completed else None)
    return task


async def _wait_for_ingestion(repository: IngestionJobRepository, context: dict,
                              job_id: str) -> dict:
    while True:
        job = await asyncio.to_thread(repository.get, context["workspace_id"],
                                      context["principal_id"], job_id)
        if job["status"] in ("ready", "failed", "cancelled"):
            return job
        task = _ensure_ingestion_task(job_id)
        await asyncio.shield(task)
        if task.done():
            await asyncio.sleep(0.05)


def _ingestion_response(job: dict, request_id: str):
    if job["status"] == "ready":
        return job["result"]
    if job["status"] == "cancelled":
        return error_response(409, "ingestion_cancelled", "Ingestion was cancelled before publication.", request_id)
    error = job["error"] or {"http_status": 500, "code": "ingestion_failed",
                             "message": "Ingestion failed before publication."}
    return error_response(error["http_status"], error["code"], error["message"], request_id)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_files(request: Request):
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data;"):
        raise HTTPException(status_code=415)
    context = request.state.request_context
    idempotency_key = request.headers.get("idempotency-key") or uuid4().hex
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{8,128}", idempotency_key):
        raise HTTPException(status_code=422)
    inputs: list[tuple[str, bytes]] = []
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
                inputs.append((name, bytes(content)))
    except StarletteHTTPException as exc:
        if exc.status_code == 400 and "Too many files" in str(exc.detail):
            raise HTTPException(status_code=413) from None
        raise

    workspace_repository = get_repository()
    previous = await asyncio.to_thread(workspace_repository.state, context["workspace_id"])
    jobs = get_ingestion_repository()
    job = await asyncio.to_thread(jobs.submit, context["workspace_id"], context["principal_id"],
                                  idempotency_key, inputs, previous["active_version"])
    if job["status"] not in ("ready", "failed", "cancelled"):
        _ensure_ingestion_task(job["job_id"])
    if "respond-async" in request.headers.get("prefer", "").lower():
        return JSONResponse({"job": job}, status_code=202)
    job = await _wait_for_ingestion(jobs, context, job["job_id"])
    return _ingestion_response(job, request.state.request_id)


@app.get("/api/ingestion-jobs")
async def list_ingestion_jobs(request: Request, limit: int = Query(20, ge=1, le=100)):
    context = request.state.request_context
    repository = await asyncio.to_thread(get_ingestion_repository)
    await asyncio.to_thread(repository.recover_expired)
    jobs = await asyncio.to_thread(repository.list_recent, context["workspace_id"],
                                   context["principal_id"], limit)
    for job in jobs:
        if job["status"] == "queued":
            _ensure_ingestion_task(job["job_id"])
    return {"jobs": jobs}


@app.get("/api/ingestion-jobs/{job_id}")
async def get_ingestion_job(job_id: str, request: Request):
    context = request.state.request_context
    repository = await asyncio.to_thread(get_ingestion_repository)
    await asyncio.to_thread(repository.recover_expired)
    job = await asyncio.to_thread(repository.get, context["workspace_id"],
                                  context["principal_id"], job_id)
    if job["status"] == "queued":
        _ensure_ingestion_task(job_id)
    return job


@app.delete("/api/ingestion-jobs/{job_id}")
async def cancel_ingestion_job(job_id: str, request: Request):
    context = request.state.request_context
    return await asyncio.to_thread(lambda: get_ingestion_repository().request_cancel(
        context["workspace_id"], context["principal_id"], job_id
    ))


def _run_chat_graph(req: ChatRequest, context: dict, history: list[dict], conversation_id: str,
                    idempotency_key: str) -> dict:
    snapshot = get_repository().snapshot(context["workspace_id"])
    if needs_reference_clarification(req.query, history):
        return {
            "conversation_id": conversation_id,
            "idempotency_key": idempotency_key,
            "workspace_id": snapshot.workspace_id,
            "corpus_version": snapshot.version,
            "response": "Please name the document, policy, or subject you mean so I can retrieve the correct evidence.",
            "source": "",
            "sources": [],
            "documents": [],
            "retrieval_status": {"status": "not_run", "reason": "unresolved_reference"},
            "citations": [],
            "evidence_status": {"status": "not_evaluated"},
            "response_plan": {},
            "result_status": "needs_clarification",
            "tool_result": None,
        }
    episodic_context = get_episodic_repository().recall(
        context["workspace_id"], context["principal_id"], req.query, snapshot.version
    )
    semantic_context = get_semantic_repository().recall(
        context["workspace_id"], context["principal_id"], req.query,
    )
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
            "episodic_context": episodic_context,
            "semantic_context": semantic_context,
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
        "episodic_context": episodic_context,
        "semantic_context": semantic_context,
    }

    try:
        result = _graph.invoke(graph_state)
    except (LLMConfigurationError, OllamaModelMemoryError):
        raise ProviderUnavailableError() from None

    return {
        "conversation_id": conversation_id,
        "idempotency_key": idempotency_key,
        "workspace_id": snapshot.workspace_id,
        "corpus_version": snapshot.version,
        "response": result.get("response", "No answer generated."),
        "source": result.get("source", ""),
        "sources": result.get("sources", []),
        "documents": [_doc_to_dict(d) for d in (result.get("documents") or [])],
        "retrieval_status": result.get("retrieval_status"),
        "citations": result.get("citations", []),
        "evidence_status": result.get("evidence_status"),
        "response_plan": result.get("response_plan", {}),
        "result_status": result.get("result_status"),
        "tool_result": result.get("tool_result"),
    }


def _chat_deadline_seconds() -> float:
    try:
        deadline = float(os.getenv("DOCQA_CHAT_DEADLINE_SECONDS", "120"))
    except ValueError as exc:
        raise RuntimeError("DOCQA_CHAT_DEADLINE_SECONDS must be numeric") from exc
    if not 1 <= deadline <= 600:
        raise RuntimeError("DOCQA_CHAT_DEADLINE_SECONDS must be between 1 and 600")
    return deadline


async def _run_bounded_chat(req: ChatRequest, context: dict, history: list[dict],
                            conversation_id: str, idempotency_key: str) -> dict:
    deadline = _chat_deadline_seconds()
    try:
        await asyncio.wait_for(_CHAT_SLOTS.acquire(), timeout=1.0)
    except TimeoutError:
        raise ChatCapacityError() from None
    task = asyncio.create_task(asyncio.to_thread(_run_chat_graph, req, context, history,
                                                  conversation_id, idempotency_key))
    release_here = True
    try:
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=deadline)
        except TimeoutError:
            release_here = False
            asyncio.create_task(_release_chat_slot_when_done(task))
            raise ChatDeadlineError() from None
        except asyncio.CancelledError:
            release_here = False
            asyncio.create_task(_release_chat_slot_when_done(task))
            raise
    finally:
        if release_here:
            _CHAT_SLOTS.release()


async def _release_chat_slot_when_done(task: asyncio.Task) -> None:
    try:
        await task
    except BaseException:
        pass
    finally:
        _CHAT_SLOTS.release()


async def _maintain_turn_lease(repository: ConversationRepository, context: dict,
                               conversation_id: str, idempotency_key: str,
                               lease_token: str, lease_lost: asyncio.Event) -> None:
    interval = max(1, repository.lease_seconds // 3)
    while True:
        await asyncio.sleep(interval)
        try:
            await asyncio.to_thread(repository.renew, context["workspace_id"], context["principal_id"],
                                    conversation_id, idempotency_key, lease_token)
        except Exception:
            lease_lost.set()
            return


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
    if reserved.lease_token is None:
        raise ConversationConflict()
    lease_lost = asyncio.Event()
    heartbeat = asyncio.create_task(_maintain_turn_lease(
        repository, context, reserved.conversation_id, idempotency_key,
        reserved.lease_token, lease_lost
    ))
    try:
        response = await _run_bounded_chat(req, context, reserved.history,
                                           reserved.conversation_id, idempotency_key)
        if lease_lost.is_set():
            raise ConversationConflict()
        await asyncio.to_thread(repository.complete, context["workspace_id"], context["principal_id"],
                                reserved.conversation_id, idempotency_key, reserved.lease_token, response)
        return response
    except BaseException:
        await asyncio.to_thread(repository.fail, context["workspace_id"], context["principal_id"],
                                reserved.conversation_id, idempotency_key, reserved.lease_token)
        raise
    finally:
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat


@app.get("/api/conversations")
async def list_conversations(request: Request):
    context = request.state.request_context
    conversations = await asyncio.to_thread(get_conversation_repository().list_recent,
                                             context["workspace_id"], context["principal_id"])
    return {"conversations": conversations}


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, request: Request,
                           limit: int = Query(50, ge=1, le=100),
                           before_sequence: int | None = Query(None, ge=1)):
    context = request.state.request_context
    return await asyncio.to_thread(get_conversation_repository().get,
                                   context["workspace_id"], context["principal_id"], conversation_id,
                                   limit=limit, before_sequence=before_sequence)


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, request: Request):
    context = request.state.request_context
    await asyncio.to_thread(get_conversation_repository().delete,
                            context["workspace_id"], context["principal_id"], conversation_id)
    return {"deleted": True, "conversation_id": conversation_id}


@app.post("/api/episodes")
async def save_episode(req: EpisodeFeedbackRequest, request: Request):
    context = request.state.request_context
    return await asyncio.to_thread(
        get_episodic_repository().save_feedback,
        context["workspace_id"], context["principal_id"], req.conversation_id,
        req.idempotency_key, req.correction,
    )


@app.get("/api/episodes")
async def list_episodes(request: Request, limit: int = Query(50, ge=1, le=100)):
    context = request.state.request_context
    episodes = await asyncio.to_thread(
        get_episodic_repository().list_recent,
        context["workspace_id"], context["principal_id"], limit=limit,
    )
    return {"episodes": episodes}


@app.delete("/api/episodes/{episode_id}")
async def delete_episode(episode_id: str, request: Request):
    context = request.state.request_context
    await asyncio.to_thread(
        get_episodic_repository().delete,
        context["workspace_id"], context["principal_id"], episode_id,
    )
    return {"deleted": True, "episode_id": episode_id}


@app.put("/api/memories/semantic")
async def save_semantic_memory(req: SemanticMemoryRequest, request: Request):
    context = request.state.request_context
    return await asyncio.to_thread(
        get_semantic_repository().upsert,
        context["workspace_id"], context["principal_id"], req.category, req.key, req.value,
    )


@app.get("/api/memories/semantic")
async def list_semantic_memories(request: Request, limit: int = Query(50, ge=1, le=100)):
    context = request.state.request_context
    memories = await asyncio.to_thread(
        get_semantic_repository().list_recent,
        context["workspace_id"], context["principal_id"], limit=limit,
    )
    return {"memories": memories}


@app.get("/api/memories/semantic/export")
async def export_semantic_memories(request: Request):
    context = request.state.request_context
    return await asyncio.to_thread(
        get_semantic_repository().export,
        context["workspace_id"], context["principal_id"],
    )


@app.delete("/api/memories/semantic/{memory_id}")
async def delete_semantic_memory(memory_id: str, request: Request):
    context = request.state.request_context
    await asyncio.to_thread(
        get_semantic_repository().delete,
        context["workspace_id"], context["principal_id"], memory_id,
    )
    return {"deleted": True, "memory_id": memory_id}


@app.get("/api/state")
async def get_state(request: Request):
    workspace = request.state.request_context["workspace_id"]
    return await asyncio.to_thread(lambda: get_repository().describe(workspace))


@app.get("/api/workspaces")
async def get_workspaces(request: Request):
    context = request.state.request_context
    return {"workspace_id": context["workspace_id"], "workspaces": context["allowed_workspaces"],
            "can_select": context["can_select_workspace"]}


@app.post("/api/provider")
async def set_provider(req: ProviderRequest, request: Request):
    workspace = request.state.request_context["workspace_id"]
    await asyncio.to_thread(lambda: get_repository().set_provider(workspace, req.provider))
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
