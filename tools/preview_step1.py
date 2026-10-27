"""Local synthetic UI preview; no real models, credentials, or saved user corpus.

Run: python3 -m tools.preview_step1 --port 8765
Open /_checks for browser sanitizer fixtures or / for the real UI.
"""

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workspaces", action="store_true", help="Preview synthetic alpha/beta membership on loopback only")
    args = parser.parse_args()
    os.environ.update({"DOCQA_AUTH_MODE": "local", "OMP_NUM_THREADS": "1", "PYTHON_DOTENV_DISABLED": "1",
                       "GOOGLE_API_KEY": "", "TAVILY_API_KEY": "", "HF_HUB_OFFLINE": "1",
                       "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false", "LLM_PROVIDER": "ollama"})
    from fastapi.responses import FileResponse
    import uvicorn
    import app.server as server
    from langchain_core.embeddings import Embeddings
    import vector_Store.faiss_store as store

    class SyntheticEmbeddings(Embeddings):
        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_query(text) for text in texts]

    @server.app.get("/_checks")
    async def browser_checks():
        return FileResponse(Path(__file__).resolve().parents[1] / "tests/browser_security.html")

    with ExitStack() as stack:
        folder = stack.enter_context(tempfile.TemporaryDirectory())
        stack.enter_context(patch.object(store, "DB_PATH", Path(folder)))
        from storage.workspaces import WorkspaceRepository
        stack.enter_context(patch.object(server, "get_repository", side_effect=lambda: WorkspaceRepository(Path(folder), SyntheticEmbeddings, "ollama")))
        stack.enter_context(patch.object(store, "get_cached_embedding", return_value=SyntheticEmbeddings()))
        # Explicit test-only identity adapter. Local host/peer/origin checks remain active.
        if args.workspaces:
            from app.security import authorize, BoundaryError
            from starlette.datastructures import Headers
            from storage.workspaces import WorkspaceRepository
            from langchain_core.documents import Document
            def preview_authorize(scope, headers, settings):
                clean = Headers(raw=[(k, v) for k, v in headers.raw if k.lower() != b"x-workspace-id"])
                context = authorize(scope, clean, settings)
                workspace = headers.get("x-workspace-id", "alpha")
                if workspace not in ("alpha", "beta"):
                    raise BoundaryError(403, "workspace_forbidden", "Unknown preview workspace")
                return {**context, "workspace_id": workspace, "allowed_workspaces": ["alpha", "beta"], "can_select_workspace": True}
            stack.enter_context(patch("app.security.authorize", side_effect=preview_authorize))
            repository = WorkspaceRepository(Path(folder), SyntheticEmbeddings, "ollama")
            for workspace in ("alpha", "beta"):
                vectorstore = store.build_vector_store([Document(page_content=f"{workspace} preview policy", metadata={"source": f"{workspace}.txt"})])
                repository.publish(workspace, vectorstore, [], [f"What is the {workspace} policy?"], ["txt"],
                                   [{"name": f"{workspace}.txt", "sha256": "synthetic-preview"}], expected_version=None)

        stack.enter_context(patch("retrieval.reranker.get_reranker_model", return_value=None))
        stack.enter_context(patch.object(server, "generate_questions", return_value=["What is the alpha deadline?"]))
        stack.enter_context(patch("utils.prompt_skill.get_prompt_skill", return_value={"body": ""}))
        for module, text in (("intent_classifier", "document_qa"), ("response_planner", "intent: document_qa"),
                             ("qa_agent", '**Synthetic preview:** alpha returns allow **30 days**. <img src=x onerror="window.securitySentinel=1">')):
            stack.enter_context(patch(f"agents.{module}.invoke_orchestrator", return_value=SimpleNamespace(content=text)))
        # Bound all tool routes in this preview, including the UI's web toggle.
        stack.enter_context(patch("langgraph_flow.graph_builder.web_search_agent", side_effect=lambda state: {**state, "documents": [], "response": "Synthetic preview: web access is disabled."}))
        uvicorn.run(server.app, host="127.0.0.1", port=args.port, proxy_headers=False)


if __name__ == "__main__":
    main()
