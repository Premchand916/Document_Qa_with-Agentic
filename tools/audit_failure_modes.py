"""Reproduce failure modes with synthetic data and no network/model downloads.

Run from the repository root:
    python -m tools.audit_failure_modes --output docs/failure-mode-evidence.json

This is a diagnostic harness, not a passing regression suite. A 'reproduced'
result confirms an undesirable behavior; it does not mean the app is safe.
Exit 1 means a probe had an unexpected error. Index writes use temporary folders.
"""

import argparse
import asyncio
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import app.server as server
import agents.intent_classifier as intent
import agents.qa_agent as qa
import agents.response_planner as response_planner
import agents.retriever_agent as retrieval
from agents.dataframe_tool import analyze_dataframe, dataframe_tool_agent
from agents.planner_agent import planner_agent
import embeddings.Hugging_face_embedding as embedding_module
from ingestion.file_loader import _normalize_dataframe
from ingestion.semantic_chunker import semantic_chunk_documents
import langgraph_flow.graph_builder as graph_builder
from memory.conversation_memory import get_memory
import retrieval.reranker as reranker
import utils.prompt_skill as prompt_skill
import vector_Store.faiss_store as store_module


CASES = []


def probe(identifier, title, evidence):
    def register(function):
        CASES.append((identifier, title, evidence, function))
        return function
    return register


class TinyEmbeddings(Embeddings):
    """Synthetic coordinates test persistence contracts, not semantic quality."""

    def __init__(self, dimension=2, reverse=False):
        self.dimension = dimension
        self.reverse = reverse

    def embed_query(self, text):
        first = ("alpha" in text.lower()) != self.reverse
        return ([1.0, 0.0] if first else [0.0, 1.0]) + [0.0] * (self.dimension - 2)

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


def documents():
    return [Document(page_content="alpha policy", metadata={"source": "alpha.txt"}),
            Document(page_content="beta policy", metadata={"source": "beta.txt"})]


def asset(name, value):
    return {"source": name, "sheet": "Data", "file_type": "csv",
            "dataframe": pd.DataFrame({"amount": [value]})}


@contextmanager
def api(graph=None):
    """Run real handlers with an isolated persistent repository and fake embeddings."""
    from storage.workspaces import WorkspaceRepository
    from app.security import AccessSettings
    with tempfile.TemporaryDirectory() as folder, \
            patch.object(server, "get_repository", side_effect=lambda: WorkspaceRepository(Path(folder), TinyEmbeddings, "ollama")), \
            patch.object(store_module, "get_cached_embedding", return_value=TinyEmbeddings()), \
            patch.object(server, "_graph", graph or SimpleNamespace(invoke=lambda s: s)), \
            patch.object(prompt_skill, "get_prompt_skill", return_value={"body": ""}), \
            patch("app.security.get_access_settings", return_value=AccessSettings(mode="local")):
        client = TestClient(server.app, raise_server_exceptions=False, client=("127.0.0.1", 12345), base_url="http://127.0.0.1")
        try:
            yield client
        finally:
            client.close()


def seed_api_corpus():
    repository = server.get_repository()
    repository.publish("local", store_module.build_vector_store(documents()), [], [], ["txt"], [],
                       expected_version=repository.state("local")["active_version"])


@contextmanager
def two_workspaces():
    from app.security import AccessSettings
    settings = AccessSettings(mode="auth_proxy", public_origin="https://docs.example.test",
                              proxy_secret="synthetic-proxy-secret-" * 3,
                              workspace_memberships={"alpha": ("alice",), "beta": ("bob",)})
    with patch("app.security.get_access_settings", return_value=settings):
        yield [{"host": "docs.example.test", "x-docqa-proxy-secret": settings.proxy_secret,
                "x-auth-request-user": user, "x-workspace-id": workspace}
               for user, workspace in (("alice", "alpha"), ("bob", "beta"))]


@contextmanager
def graph_models(qa_result="synthetic answer"):
    with patch.object(intent, "invoke_orchestrator", return_value=SimpleNamespace(content="document_qa")) as router, \
            patch.object(response_planner, "invoke_orchestrator", return_value=SimpleNamespace(content="intent: document_qa")) as planner, \
            patch.object(qa, "invoke_orchestrator", return_value=SimpleNamespace(content=qa_result)) as answer:
        yield (router, planner, answer)


@probe("M01", "Conversation history disappears without a prompt profile", "utils/prompt_skill.py:107")
def history_without_profile():
    state = {"chat_history": [{"role": "user", "content": "AUDIT_TOPIC_ALPHA"}]}
    with patch.object(prompt_skill, "get_prompt_skill", return_value={"body": ""}):
        result = prompt_skill.build_prompted_query("What about its deadline?", state)
    return "AUDIT_TOPIC_ALPHA" not in result, {"prompt": result}


@probe("M02", "Malformed history role produces HTTP 500 when profile is enabled", "utils/prompt_skill.py:97")
def malformed_history():
    with api() as client, patch.object(prompt_skill, "get_prompt_skill", return_value={"body": "audit profile"}):
        seed_api_corpus()
        result = client.post("/api/chat", json={"query": "hello", "chat_history": [{"role": None, "content": "x"}]})
    return result.status_code == 500, {"status": result.status_code}


@probe("M03", "Memory factory does not recover a previous conversation", "memory/conversation_memory.py:51")
def fresh_memory():
    first = get_memory()
    first.add_user_message("remember alpha")
    second = get_memory()
    return not second.messages, {"first_messages": len(first.messages), "new_instance_messages": len(second.messages)}


@probe("M04", "Returned memory messages allow mutation of the stored history", "memory/conversation_memory.py:17")
def shallow_memory_copy():
    memory = get_memory()
    memory.add_user_message("original")
    memory.load_memory_variables()["chat_history"][0]["content"] = "changed"
    return memory.messages[0]["content"] == "changed", {"stored_content": memory.messages[0]["content"]}


@probe("A01", "A CSV routes an unrelated PDF question to the table tool", "agents/planner_agent.py:14")
def mixed_routing():
    state = planner_agent({"query": "What is the PDF cancellation policy?", "intent": "document_qa",
                           "tabular_assets": [asset("sales.csv", 10)], "uploaded_file_types": ["PDF", "CSV"]})
    return state["use_tool"], {"use_tool": state["use_tool"]}


@probe("A02", "A question about the second table computes on the first table", "agents/dataframe_tool.py:76")
def wrong_table():
    result = dataframe_tool_agent({"query": "Total amount in second.csv", "tabular_assets": [asset("first.csv", 10), asset("second.csv", 99)]})
    return result["source"].startswith("first.csv"), {"source": result["source"], "answer": result["response"], "expected_amount": 99}


@probe("A03", "A filtered total silently includes excluded rows", "agents/dataframe_tool.py:18")
def ignored_filter():
    frame = pd.DataFrame({"region": ["North", "South"], "amount": [10, 20]})
    result = analyze_dataframe(frame, "Total amount for North only")
    return result["total"]["amount"] != 10, {"expected": 10, "actual": int(result["total"]["amount"])}


@probe("A04", "Mixed upload reports success after index failure, then chat crashes", "app/server.py:117")
def partial_upload():
    with graph_models(), api(graph_builder.build_graph()) as client, \
            patch.object(server, "build_vector_store", side_effect=RuntimeError("synthetic embedding failure")), \
            patch.object(server, "generate_questions", return_value=[]):
        upload = client.post("/api/upload", files=[("files", ("policy.txt", b"alpha policy")), ("files", ("sales.csv", b"amount\n10\n"))])
        chat = client.post("/api/chat", json={"query": "What does the policy say?"})
    return upload.status_code == 200 and chat.status_code == 500, {"upload_status": upload.status_code, "readiness": upload.json(), "chat_status": chat.status_code}


@probe("A05", "Response planning is not applied to the final answer prompt", "agents/qa_agent.py:40")
def ignored_plan():
    with patch.object(qa, "invoke_orchestrator", return_value=SimpleNamespace(content="answer")) as model:
        qa.qa_agent({"query": "Summarize", "documents": documents(), "response_plan": {"sections": ["AUDIT_REQUIRED_SECTION"]}})
        prompt = model.call_args.args[0]
    return "AUDIT_REQUIRED_SECTION" not in prompt, {"planned_section_in_prompt": "AUDIT_REQUIRED_SECTION" in prompt}


@probe("A06", "Blank numeric cells make a numeric column disappear from totals", "ingestion/file_loader.py:53")
def numeric_dtype_loss():
    frame = _normalize_dataframe(pd.DataFrame({"amount": [10.0, None, 30.0]}))
    result = analyze_dataframe(frame, "total amount")
    return "amount" not in result["total"], {"expected_amount": 40, "actual": result, "dtype_after_normalization": str(frame.dtypes["amount"])}


@probe("R01", "Fusion collapses identical text with different source provenance", "agents/retriever_agent.py:44")
def merged_sources():
    result = retrieval._rrf_fusion([[Document(page_content="same", metadata={"source": "a"})], [Document(page_content="same", metadata={"source": "b"})]])
    return len(result) == 1, {"input_sources": 2, "retained_sources": len(result)}


@probe("R02", "BM25 loses case and punctuation variants", "agents/retriever_agent.py:37")
def lexical_normalization():
    docs = tuple(Document(page_content=t) for t in ["Refund eligibility", "Solar panels", "River levels"])
    bm25, _ = retrieval._get_or_build_bm25(123456, docs)
    exact = float(bm25.get_scores(["Refund"])[0])
    variant = float(bm25.get_scores(["refund?"])[0])
    return exact > 0 and variant == 0, {"exact_score": exact, "variant_score": variant}


@probe("R03", "Zero lexical overlap and no vector hits still return evidence", "agents/retriever_agent.py:78")
def irrelevant_evidence():
    docs = [Document(page_content=t) for t in ["solar panel output", "river water level", "garden rose petals"]]
    vs = SimpleNamespace(docstore=SimpleNamespace(_dict=dict(enumerate(docs))), similarity_search=lambda query, k: [])
    with patch.object(retrieval, "rerank_documents", side_effect=lambda q, ds, top_k: ds[:top_k]):
        result = retrieval.retriever_agent({"query": "xylophone warranty", "vectorstore": vs})
    return bool(result["documents"]), {"vector_hits": 0, "lexical_overlap": 0, "returned_chunks": len(result["documents"])}


@probe("R04", "A failed reranker load is cached until cache clear or restart", "retrieval/reranker.py:29")
def cached_reranker_failure():
    reranker.get_reranker_model.cache_clear()
    try:
        with patch.object(reranker, "CrossEncoder", side_effect=[RuntimeError("model unavailable"), object()]) as loader:
            first = reranker.get_reranker_model()
            second = reranker.get_reranker_model()
        return first is None and second is None and loader.call_count == 1, {"loader_calls": loader.call_count, "second_result_is_none": second is None}
    finally:
        reranker.get_reranker_model.cache_clear()


@probe("I01", "Large JSON bypasses the text chunk size", "ingestion/semantic_chunker.py:15")
def oversized_json():
    content = json.dumps({"rows": ["alpha " * 100] * 30})
    result = semantic_chunk_documents([Document(page_content=content, metadata={"content_type": "json_document"})])
    longest = max(len(doc.page_content) for doc in result)
    return len(result) == 1 and longest > 10000, {"chunks": len(result), "longest_characters": longest}


@probe("E01", "Different embedding dimensions load successfully then fail on search", "vector_Store/faiss_store.py:45")
def dimension_mismatch():
    from storage.workspaces import CorpusIntegrityError
    with tempfile.TemporaryDirectory() as folder, patch.object(store_module, "DB_PATH", Path(folder)), \
            patch.object(store_module, "get_cached_embedding", return_value=TinyEmbeddings()):
        store_module.create_vector_store(documents())
        with patch.object(store_module, "get_cached_embedding", return_value=TinyEmbeddings(dimension=3)):
            try:
                loaded = store_module.load_vector_store_direct()
            except CorpusIntegrityError:
                return False, {"load_rejected": True, "index_dimension": 2, "query_dimension": 3}
            try:
                loaded.similarity_search("alpha", k=1)
            except (AssertionError, ValueError, RuntimeError) as exc:
                return True, {"load_succeeded": True, "index_dimension": loaded.index.d, "query_dimension": 3, "search_error": type(exc).__name__}
    return False, {"load_succeeded": True, "search_error": None}


@probe("E02", "Same-dimensional incompatible embeddings silently return the wrong document", "vector_Store/faiss_store.py:45")
def coordinate_mismatch():
    from storage.workspaces import CorpusIntegrityError
    with tempfile.TemporaryDirectory() as folder, patch.object(store_module, "DB_PATH", Path(folder)), \
            patch.object(store_module, "get_cached_embedding", return_value=TinyEmbeddings()):
        original = store_module.create_vector_store(documents())
        expected = original.similarity_search("alpha", k=1)[0].metadata["source"]
        with patch.object(store_module, "get_cached_embedding", return_value=TinyEmbeddings(reverse=True)):
            try:
                actual = store_module.load_vector_store_direct().similarity_search("alpha", k=1)[0].metadata["source"]
            except CorpusIntegrityError:
                return False, {"load_rejected": True, "expected_source": expected}
    return actual != expected, {"expected_source": expected, "actual_source": actual, "exception": None}


@probe("E03", "Changing EMBEDDING_MODEL does not invalidate the model cache", "embeddings/Hugging_face_embedding.py:32")
def embedding_cache():
    embedding_module.get_embedding_model.cache_clear()
    try:
        with patch.object(embedding_module, "HuggingFaceEmbeddings", side_effect=lambda **kw: SimpleNamespace(name=kw["model_name"])):
            with patch.dict(os.environ, {"EMBEDDING_MODEL": "audit-first"}):
                first = embedding_module.get_embedding_model()
            with patch.dict(os.environ, {"EMBEDDING_MODEL": "audit-second"}):
                second = embedding_module.get_embedding_model()
        return first is second, {"requested_second": "audit-second", "actual_second": second.name}
    finally:
        embedding_module.get_embedding_model.cache_clear()


@probe("S01", "An independent client can query a previous client's uploaded document", "app/server.py:50")
def shared_workspace():
    graph = SimpleNamespace(invoke=lambda s: {"response": s["vectorstore"].similarity_search("alpha", k=1)[0].page_content})
    with api(graph) as client, two_workspaces() as headers, patch.object(server, "generate_questions", return_value=[]):
        upload = client.post("/api/upload", headers=headers[0], files={"files": ("private.txt", b"AUDIT_CLIENT_A_ONLY")})
        own = client.post("/api/chat", headers=headers[0], json={"query": "Show the document"})
        response = client.post("/api/chat", headers=headers[1], json={"query": "Show the document"})
        assert upload.status_code == 200 and "AUDIT_CLIENT_A_ONLY" in own.json()["response"]
    return response.status_code == 200 and "AUDIT_CLIENT_A_ONLY" in response.json().get("response", ""), {"owner_status": own.status_code, "second_workspace_status": response.status_code, "other_workspace_marker_returned": "AUDIT_CLIENT_A_ONLY" in response.json().get("response", "")}


@probe("S02", "Provider changes affect another workspace", "app/server.py:set_provider")
def shared_provider():
    with api() as client, two_workspaces() as headers:
        assert client.post("/api/provider", headers=headers[0], json={"provider": "gemini"}).status_code == 200
        provider = client.get("/api/state", headers=headers[1]).json()["provider"]
        owner = client.get("/api/state", headers=headers[0]).json()["provider"]
    return provider == "gemini", {"owner_provider": owner, "second_workspace_provider": provider}


@probe("P01", "Restart restores vectors but loses spreadsheet computation state", "app/server.py:_startup")
def restart_loses_tables():
    with api() as client, patch.object(server, "generate_questions", return_value=[]):
        response = client.post("/api/upload", files={"files": ("sales.csv", b"amount\n10\n")})
        before = len(server.get_repository().snapshot("local").tabular_assets)
        asyncio.run(server._startup())
        restored = server.get_repository().snapshot("local")
        facts = {"upload_status": response.status_code, "tables_before_restart": before,
                 "vectorstore_restored": restored.vectorstore is not None, "tabular_assets_restored": len(restored.tabular_assets)}
    return before == 1 and facts["vectorstore_restored"] and facts["tabular_assets_restored"] == 0, facts


@probe("P02", "Blocking graph invocation stalls the API event loop", "app/server.py:184")
def event_loop_block():
    def slow_invoke(state):
        time.sleep(0.06)
        return {"response": "done"}

    async def run():
        start = time.perf_counter()

        async def tick():
            await asyncio.sleep(0.005)
            return time.perf_counter() - start

        task = asyncio.create_task(tick())
        await asyncio.sleep(0)
        await server.chat(server.ChatRequest(query="hello"), SimpleNamespace(state=SimpleNamespace(request_context={"workspace_id": "local", "principal_id": "local-developer"})))
        return await task

    with api(SimpleNamespace(invoke=slow_invoke)):
        seed_api_corpus()
        delay = asyncio.run(run())
    return delay >= 0.05, {"requested_timer_ms": 5, "observed_timer_ms": round(delay * 1000, 1), "synthetic_block_ms": 60}


@probe("P03", "LLM configuration failure is returned as a successful answer", "agents/qa_agent.py:83")
def success_shaped_error():
    with graph_models(), patch.object(graph_builder, "retriever_agent", side_effect=lambda s: {**s, "documents": documents()}):
        graph = graph_builder.build_graph()
        with api(graph) as client, patch.object(qa, "invoke_orchestrator", side_effect=qa.LLMConfigurationError("AUDIT_PROVIDER_UNAVAILABLE")):
            seed_api_corpus()
            response = client.post("/api/chat", json={"query": "alpha policy"})
    return response.status_code == 200 and "AUDIT_PROVIDER_UNAVAILABLE" in response.json()["response"], {"status": response.status_code, "response": response.json().get("response")}


@probe("P04", "Interrupted index save destroys the previously loadable corpus", "vector_Store/faiss_store.py:39")
def interrupted_save():
    with tempfile.TemporaryDirectory() as folder, patch.object(store_module, "DB_PATH", Path(folder)), \
            patch.object(store_module, "get_cached_embedding", return_value=TinyEmbeddings()):
        store_module.create_vector_store(documents())
        before = store_module.load_vector_store_direct().index.ntotal
        try:
            with patch("storage.workspaces.write_json", side_effect=OSError("synthetic disk-full error")):
                store_module.create_vector_store(documents()[:1])
        except OSError:
            pass
        try:
            restored = store_module.load_vector_store_direct()
            if restored.index.ntotal != before:
                return True, {"prior_index_vectors": before, "restored_vectors": restored.index.ntotal}
        except Exception as exc:
            return True, {"prior_index_vectors": before, "reload_error": type(exc).__name__, "fault": "injected disk write failure in temporary directory"}
    return False, {"prior_index_vectors": before, "reload_error": None}


@probe("V01", "Blank query is accepted by the HTTP API", "app/server.py:65")
def blank_query():
    with api(SimpleNamespace(invoke=lambda s: {"response": "called"})) as client:
        seed_api_corpus()
        response = client.post("/api/chat", json={"query": ""})
    return response.status_code == 200, {"status": response.status_code}


@probe("V02", "Unknown file types are rejected (control)", "ingestion/file_loader.py:325")
def unsupported_upload_control():
    with api() as client:
        response = client.post("/api/upload", files={"files": ("audit.exe", b"not executable")})
    # For this control only, reproduced means the intended protection worked.
    return response.status_code == 422, {"status": response.status_code, "control": True}


def trace_workflows():
    traces = {}
    for name, extra in (("document", {}), ("table", {"tabular_assets": [asset("sales.csv", 10)], "uploaded_file_types": ["CSV"]}), ("web", {"use_web_search": True})):
        with graph_models() as models, \
                patch.object(graph_builder, "retriever_agent", side_effect=lambda s: {**s, "documents": documents()}), \
                patch.object(graph_builder, "web_search_agent", side_effect=lambda s: {**s, "documents": documents()}):
            graph = graph_builder.build_graph()
            updates = list(graph.stream({"query": "alpha policy", "vectorstore": object(), **extra}, stream_mode="updates"))
            traces[name] = {"executed_nodes": [node for update in updates for node in update], "model_calls": sum(model.call_count for model in models), "checkpointer_configured": graph.checkpointer is not None}
    return traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []
    with ExitStack() as guards:
        guards.enter_context(patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "HF_LOCAL_FILES_ONLY": "true", "GOOGLE_API_KEY": "", "TAVILY_API_KEY": ""}))
        guards.enter_context(patch.object(socket.socket, "connect", side_effect=RuntimeError("Network disabled during audit")))
        guards.enter_context(patch.object(socket, "create_connection", side_effect=RuntimeError("Network disabled during audit")))
        for identifier, title, evidence, function in CASES:
            retrieval.clear_retrieval_caches()
            try:
                reproduced, observed = function()
                status = "reproduced" if reproduced else "not_reproduced"
                if observed.get("control"):
                    status = "control_passed" if reproduced else "control_failed"
                results.append({"id": identifier, "title": title, "status": status, "source": evidence, "observed": observed})
            except Exception as exc:
                results.append({"id": identifier, "title": title, "status": "probe_error", "source": evidence, "error_type": type(exc).__name__, "error": str(exc)[:250]})
        retrieval.clear_retrieval_caches()
        workflow_traces = trace_workflows()
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Synthetic offline probes; model calls stubbed; real handlers, graph and FAISS persistence where described; no production data",
        "workflow_traces": workflow_traces,
        "versions": {p: importlib.metadata.version(p) for p in ("fastapi", "langgraph", "faiss-cpu", "pandas", "sentence-transformers", "langchain-community", "langchain-huggingface")},
        "summary": {status: sum(r["status"] == status for r in results) for status in ("reproduced", "not_reproduced", "control_passed", "control_failed", "probe_error")},
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(json.dumps(report["summary"]))
        print(f"Evidence saved to {args.output}")
    else:
        print(rendered)
    return 1 if report["summary"]["probe_error"] or report["summary"]["control_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
