"""Run the real document graph on a label-blind frozen Ollama holdout set."""

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

from agents.orchestrator_agent import capture_model_calls
from embeddings.identity import embedding_identity
from ingestion.file_loader import parse_uploaded_file
from ingestion.semantic_chunker import semantic_chunk_documents
from langgraph_flow.graph_builder import build_graph
from memory.followups import resolve_followup_query
from tools.evaluation_dataset import corpus_fixture_sha256, load_inference_manifest, source_paths
from utils.prompt_skill import build_prompted_query
from vector_Store.faiss_store import build_vector_store, get_cached_embedding


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "tests/fixtures/release_holdout/tasks.json"


class _Upload(BytesIO):
    def __init__(self, name: str, content: bytes):
        super().__init__(content)
        self.name = name


def _ollama_revision(base_url: str, model: str) -> str:
    request = Request(f"{base_url.rstrip('/')}/api/tags", headers={"Accept": "application/json"})
    with urlopen(request, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    for item in payload.get("models", []):
        name = str(item.get("name") or "")
        if name == model or name.split(":", 1)[0] == model.split(":", 1)[0]:
            revision = str(item.get("digest") or "").strip()
            if revision:
                return revision
    raise RuntimeError(f"Ollama model is not installed: {model}")


def _build_corpus(tasks_path: Path, manifest: dict):
    raw_documents, table_assets = [], []
    for name, path in source_paths(tasks_path, manifest).items():
        parsed = parse_uploaded_file(_Upload(name, path.read_bytes()))
        raw_documents.extend(parsed.documents)
        table_assets.extend(parsed.tabular_assets)
    embedding = get_cached_embedding()
    tokenizer = getattr(getattr(embedding, "client", None), "tokenizer", None)
    chunks = semantic_chunk_documents(raw_documents, tokenizer=tokenizer)
    return build_vector_store(chunks), table_assets, embedding


def _abstained(result: dict) -> bool:
    if result.get("result_status") in {"insufficient_evidence", "needs_clarification", "citation_error"}:
        return True
    text = str(result.get("response") or "").casefold()
    return bool(re.search(
        r"\b(no relevant information|not (?:available|provided|found)|does not contain|"
        r"doesn't contain|insufficient (?:evidence|information)|cannot determine|can't determine)\b",
        text,
    ))


def run_evaluation(tasks_path: Path, *, model: str, base_url: str) -> dict:
    manifest = load_inference_manifest(tasks_path)
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["OLLAMA_MODEL"] = model
    os.environ["OLLAMA_BASE_URL"] = base_url
    revision = _ollama_revision(base_url, model)
    vectorstore, table_assets, embedding = _build_corpus(tasks_path, manifest)
    graph = build_graph()
    file_types = sorted({Path(name).suffix.lstrip(".") for name in manifest["sources"]})
    predictions = []
    for case in manifest["cases"]:
        history = case.get("chat_history") or []
        query = case["query"]
        common = {
            "vectorstore": vectorstore,
            "chat_history": history,
            "answer_mode": case.get("answer_mode", "Auto"),
            "audience": case.get("audience", "General"),
            "response_depth": case.get("response_depth", "Balanced"),
            "uploaded_file_types": file_types,
            "tabular_assets": table_assets,
            "use_web_search": False,
            "episodic_context": [],
        }
        state = {
            **common,
            "query": query,
            "search_query": resolve_followup_query(query, history),
            "web_search_query": resolve_followup_query(query, history),
            "prompt_query": build_prompted_query(query, common),
            "prompt_skill_path": "",
            "request_context": {"workspace_id": "frozen-holdout", "principal_id": "evaluator"},
            "provider": "ollama",
            "corpus_version": corpus_fixture_sha256(tasks_path, manifest)[:32],
        }
        started = time.monotonic()
        with capture_model_calls() as model_calls:
            result = graph.invoke(state)
        latency_ms = round((time.monotonic() - started) * 1000, 3)
        evidence_sources = list(dict.fromkeys(
            str(document.metadata.get("source"))
            for document in result.get("documents", [])
            if document.metadata.get("source")
        ))
        cited_sources = list(dict.fromkeys(
            str(citation.get("source"))
            for citation in result.get("citations", [])
            if citation.get("source")
        ))
        predictions.append({
            "id": case["id"],
            "answer": str(result.get("response") or ""),
            "evidence_sources": evidence_sources,
            "cited_sources": cited_sources,
            "abstained": _abstained(result),
            "tool_result": result.get("tool_result"),
            "latency_ms": latency_ms,
            "token_usage": {
                "input": sum(call["token_usage"]["input"] for call in model_calls),
                "output": sum(call["token_usage"]["output"] for call in model_calls),
            },
            "cost_usd": 0.0,
            "model_calls": model_calls,
            "retrieval_status": result.get("retrieval_status"),
            "resolved_query": state["search_query"],
        })
    identity = embedding_identity(embedding)
    return {
        "run": {
            "provider": "ollama",
            "model": model,
            "model_revision": revision,
            "embedding_identity": f'{identity["dimension"]}:{identity["fingerprint"]}',
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "split": "holdout",
            "corpus_fixture_sha256": corpus_fixture_sha256(tasks_path, manifest),
            "dataset_id": manifest["dataset_id"],
            "label_blind": True,
        },
        "predictions": predictions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_evaluation(args.tasks, model=args.model, base_url=args.base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(result['predictions'])} label-blind predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
