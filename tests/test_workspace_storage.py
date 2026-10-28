"""Step 2 acceptance tests: persistence, publication failure, and workspace isolation."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import threading
from types import SimpleNamespace
from unittest.mock import patch

from tests.support import OfflineCase


class WorkspaceStorageTests(OfflineCase):
    def repository(self):
        from storage.workspaces import WorkspaceRepository
        return WorkspaceRepository(self.temp / "workspaces", lambda: self.embeddings, "ollama")

    def publish(self, repository, workspace="alpha", text="alpha policy", amount=10, expected=None):
        import pandas as pd
        from langchain_core.documents import Document
        vectorstore = self.store.build_vector_store([Document(page_content=text, metadata={"source": "policy.txt"})])
        return repository.publish(workspace, vectorstore, [{"source": "sales.csv", "sheet": "Data", "file_type": "csv",
            "dataframe": pd.DataFrame({"amount": pd.Series([amount, None], dtype="Int64")})}],
            ["What is the policy?"], ["txt", "csv"], [{"name": "policy.txt", "sha256": "a" * 64}], expected_version=expected)

    def test_restart_preserves_vectors_typed_tables_sources_and_provider(self):
        repo = self.repository()
        version = self.publish(repo)
        repo.set_provider("alpha", "gemini")
        restored = self.repository().snapshot("alpha")
        self.assertEqual(restored.version, version)
        self.assertEqual(restored.provider, "gemini")
        self.assertEqual(restored.vectorstore.similarity_search("alpha", k=1)[0].page_content, "alpha policy")
        self.assertEqual(str(restored.tabular_assets[0]["dataframe"].dtypes["amount"]), "Int64")
        self.assertEqual(restored.tabular_assets[0]["dataframe"]["amount"].sum(), 10)
        self.assertEqual(restored.sources[0]["sha256"], "a" * 64)
        self.assertEqual(restored.suggested_questions, ["What is the policy?"])

    def test_extraction_warnings_survive_restart_and_replace_with_corpus(self):
        from langchain_core.documents import Document
        repo = self.repository()
        store = self.store.build_vector_store([Document(page_content="alpha", metadata={"source": "partial.pdf"})])
        report = {"source": "partial.pdf", "file_type": "pdf", "text_extracted": True,
                  "ocr_engine": None, "page_count": 2, "text_pages": 1,
                  "pages_without_text": [2], "pages_with_coordinates": [1]}
        first = repo.publish("alpha", store, [], [], ["pdf"], [{"name": "partial.pdf", "sha256": "a" * 64}],
                             warnings=["partial.pdf: page 2 needs OCR"], extraction_reports=[report], expected_version=None)
        restored = self.repository()
        self.assertEqual(restored.snapshot("alpha").warnings, ["partial.pdf: page 2 needs OCR"])
        self.assertEqual(restored.describe("alpha")["warnings"], ["partial.pdf: page 2 needs OCR"])
        self.assertEqual(restored.snapshot("alpha").extraction_reports, [report])
        self.assertEqual(restored.describe("alpha")["extraction_reports"], [report])
        restored.publish("alpha", store, [], [], ["pdf"], [], warnings=[], expected_version=first)
        self.assertEqual(self.repository().snapshot("alpha").warnings, [])
        self.assertEqual(self.repository().snapshot("alpha").extraction_reports, [])

    def test_workspace_corpora_and_providers_are_independent(self):
        repo = self.repository()
        self.publish(repo, "alpha", "alpha private", 10)
        self.publish(repo, "beta", "beta private", 99)
        repo.set_provider("alpha", "gemini")
        self.assertEqual(repo.snapshot("beta").provider, "ollama")
        self.assertEqual(repo.snapshot("beta").vectorstore.similarity_search("alpha", k=1)[0].page_content, "beta private")

    def test_failed_artifact_write_preserves_active_version(self):
        repo = self.repository()
        old = self.publish(repo)
        with patch("storage.workspaces.write_json", side_effect=OSError("synthetic disk full")):
            with self.assertRaises(OSError):
                self.publish(repo, text="replacement", expected=old)
        self.assertEqual(self.repository().snapshot("alpha").version, old)

    def test_failure_before_commit_preserves_old_pointer(self):
        repo = self.repository()
        old = self.publish(repo)
        with patch.object(repo, "_commit_publication", side_effect=OSError("synthetic commit failure")):
            with self.assertRaises(OSError):
                self.publish(repo, text="replacement", expected=old)
        self.assertEqual(self.repository().snapshot("alpha").version, old)

    def test_failure_after_commit_keeps_published_files(self):
        repo = self.repository()
        old = self.publish(repo)
        original = repo._commit_publication
        def committed_then_failed(*args):
            original(*args)
            raise OSError("synthetic lost response after commit")
        with patch.object(repo, "_commit_publication", side_effect=committed_then_failed):
            with self.assertRaises(OSError):
                self.publish(repo, text="replacement", expected=old)
        restored = self.repository().snapshot("alpha")
        self.assertNotEqual(restored.version, old)
        self.assertEqual(restored.vectorstore.similarity_search("alpha", k=1)[0].page_content, "replacement")

    def test_reader_remains_pinned_after_replacement(self):
        repo = self.repository()
        old = self.publish(repo)
        reader = repo.snapshot("alpha")
        new = self.publish(repo, text="new policy", amount=99, expected=old)
        self.assertEqual(reader.version, old)
        self.assertEqual(reader.tabular_assets[0]["dataframe"]["amount"].sum(), 10)
        self.assertEqual(reader.vectorstore.similarity_search("alpha", k=1)[0].page_content, "alpha policy")
        self.assertEqual(repo.snapshot("alpha").version, new)

    def test_concurrent_publishers_cannot_mix_artifacts(self):
        from storage.workspaces import PublicationConflict
        repo = self.repository()
        old = self.publish(repo)
        barrier = threading.Barrier(2)
        def attempt(value):
            # Use independent repository instances, as separate app workers would.
            other = self.repository()
            barrier.wait(timeout=5)
            try:
                return self.publish(other, text=f"policy {value}", amount=value, expected=old)
            except PublicationConflict:
                return "conflict"
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(attempt, [11, 22]))
        self.assertEqual(results.count("conflict"), 1)
        snapshot = repo.snapshot("alpha")
        amount = int(snapshot.tabular_assets[0]["dataframe"]["amount"].sum())
        self.assertEqual(snapshot.vectorstore.similarity_search("alpha", k=1)[0].page_content, f"policy {amount}")
        self.assertIn(snapshot.version, results)

    def test_tampering_is_rejected_before_native_index_read(self):
        from storage.workspaces import CorpusIntegrityError
        repo = self.repository()
        self.publish(repo)
        index = next((self.temp / "workspaces").rglob("index.faiss"))
        index.write_bytes(b"corrupt")
        with patch("faiss.read_index") as reader:
            with self.assertRaises(CorpusIntegrityError):
                repo.snapshot("alpha")
        reader.assert_not_called()

    def test_legacy_pickle_is_not_loaded(self):
        from storage.workspaces import LegacyIndexError
        self.store.DB_PATH.mkdir()
        (self.store.DB_PATH / "index.pkl").write_bytes(b"never deserialize")
        with patch("pickle.load") as loader:
            with self.assertRaises(LegacyIndexError):
                self.store.load_vector_store_direct()
        loader.assert_not_called()


class WorkspaceApiTests(OfflineCase):
    def setUp(self):
        super().setUp()
        from app.security import AccessSettings
        self.secret = "synthetic-proxy-secret-" * 3
        self.stack.enter_context(patch("app.security.get_access_settings", return_value=AccessSettings(
            mode="auth_proxy", public_origin="https://docs.example.test", proxy_secret=self.secret,
            workspace_memberships={"alpha": ("alice",), "beta": ("bob",)}, workspace_id="alpha")))
        self.stack.enter_context(patch.object(self.server, "generate_questions", return_value=[]))
        self.server._graph = SimpleNamespace(invoke=lambda state: {
            "response": state["vectorstore"].similarity_search("alpha", k=1)[0].page_content})

    def headers(self, user="alice", workspace="alpha"):
        return {"host": "docs.example.test", "x-auth-request-user": user,
                "x-docqa-proxy-secret": self.secret, "x-workspace-id": workspace}

    def test_other_member_cannot_read_replace_or_change_workspace(self):
        client = self.client()
        self.assertEqual(client.post("/api/upload", headers=self.headers(), files={"files": ("private.txt", b"alpha secret")}).status_code, 200)
        forged = self.headers("bob", "alpha")
        self.assertEqual(client.get("/api/state", headers=forged).status_code, 403)
        self.assertEqual(client.post("/api/chat", headers=forged, json={"query": "alpha"}).status_code, 403)
        self.assertEqual(client.post("/api/provider", headers=forged, json={"provider": "gemini"}).status_code, 403)
        self.assertEqual(client.post("/api/upload", headers=forged, files={"files": ("overwrite.txt", b"bad")}).status_code, 403)
        self.assertIn("alpha secret", client.post("/api/chat", headers=self.headers(), json={"query": "alpha"}).json()["response"])
        self.assertEqual(client.post("/api/chat", headers=self.headers("bob", "beta"), json={"query": "alpha"}).status_code, 422)

    def test_provider_change_does_not_modify_environment_or_other_workspace(self):
        before = os.environ.get("LLM_PROVIDER")
        client = self.client()
        self.assertEqual(client.post("/api/provider", headers=self.headers(), json={"provider": "gemini"}).status_code, 200)
        self.assertEqual(os.environ.get("LLM_PROVIDER"), before)
        self.assertEqual(client.get("/api/state", headers=self.headers("bob", "beta")).json()["provider"], "ollama")
        self.assertEqual(client.get("/api/state", headers=self.headers()).json()["provider"], "gemini")

    def test_failed_upload_keeps_previous_corpus(self):
        client = self.client()
        old = client.post("/api/upload", headers=self.headers(), files={"files": ("old.txt", b"alpha original")}).json()["corpus_version"]
        with patch.object(self.server, "build_vector_store", side_effect=RuntimeError("embedding unavailable")):
            response = client.post("/api/upload", headers=self.headers(), files=[("files", ("new.txt", b"replacement")), ("files", ("sales.csv", b"amount\n10\n"))])
        self.assertEqual(response.status_code, 500)
        self.assertEqual(client.get("/api/state", headers=self.headers()).json()["corpus_version"], old)
        self.assertIn("original", client.post("/api/chat", headers=self.headers(), json={"query": "alpha"}).json()["response"])

    def test_restart_restores_table_chat_without_client_filetype_hints(self):
        from langgraph_flow.graph_builder import build_graph
        self.stub_graph_models()
        client = self.client()
        self.assertEqual(client.post("/api/upload", headers=self.headers(), files={"files": ("sales.csv", b"amount\n10\n20\n")}).status_code, 200)
        asyncio.run(self.server._startup())
        self.server._graph = build_graph()
        result = client.post("/api/chat", headers=self.headers(), json={"query": "Total amount"})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertIn("amount: 30", result.json()["response"])
        self.assertTrue(client.get("/api/state", headers=self.headers()).json()["tabular_ready"])


    def test_workspace_listing_exposes_only_memberships(self):
        client = self.client()
        self.assertEqual(client.get("/api/workspaces", headers=self.headers()).json()["workspaces"], ["alpha"])
        self.assertEqual(client.get("/api/workspaces", headers=self.headers("bob", "beta")).json()["workspaces"], ["beta"])

    def test_publication_conflict_returns_409_and_preserves_corpus(self):
        from storage.workspaces import PublicationConflict, WorkspaceRepository
        client = self.client()
        old = client.post("/api/upload", headers=self.headers(), files={"files": ("old.txt", b"alpha original")}).json()["corpus_version"]
        with patch.object(WorkspaceRepository, "_commit_publication", side_effect=PublicationConflict()):
            response = client.post("/api/upload", headers=self.headers(), files={"files": ("new.txt", b"alpha replacement")})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["code"], "publication_conflict")
        self.assertEqual(client.get("/api/state", headers=self.headers()).json()["corpus_version"], old)

    def test_corrupt_corpus_fails_closed_with_safe_error(self):
        client = self.client()
        client.post("/api/upload", headers=self.headers(), files={"files": ("old.txt", b"alpha original")})
        next(self.temp.rglob("documents.json")).write_text("SECRET_CORRUPT_BYTES")
        for response in (client.get("/api/state", headers=self.headers()),
                         client.post("/api/chat", headers=self.headers(), json={"query": "alpha"})):
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["code"], "corpus_unavailable")
            self.assertNotIn("SECRET", response.text)

    def test_graph_keeps_provider_and_version_when_workspace_changes_mid_request(self):
        client = self.client()
        old = client.post("/api/upload", headers=self.headers(), files={"files": ("old.txt", b"alpha original")}).json()["corpus_version"]
        captured = []
        def invoke(state):
            self.server.get_repository().set_provider("alpha", "gemini")
            self.seed_workspace(workspace="alpha")
            captured.append(state)
            return {"response": state["vectorstore"].similarity_search("alpha", k=1)[0].page_content}
        self.server._graph = SimpleNamespace(invoke=invoke)
        response = client.post("/api/chat", headers=self.headers(), json={"query": "alpha"})
        self.assertEqual(response.json()["corpus_version"], old)
        self.assertIn("original", response.json()["response"])
        self.assertEqual(captured[0]["provider"], "ollama")
        self.assertNotEqual(client.get("/api/state", headers=self.headers()).json()["corpus_version"], old)

    def test_workspace_retrieval_does_not_retain_global_corpus_cache(self):
        import agents.retriever_agent as retrieval
        vectorstore = self.seed_workspace()
        from retrieval.reranker import RerankResult
        with patch.object(retrieval, "rerank_with_status", side_effect=lambda q, ds, top_k:
                          RerankResult(ds[:top_k], None, None, "fallback", "test")):
            result = retrieval.retriever_agent({"query": "alpha", "vectorstore": vectorstore,
                                                "request_context": {"workspace_id": "local"},
                                                "corpus_version": "test-version"})
        self.assertTrue(result["documents"])
        self.assertEqual(retrieval._VECTORSTORE_REGISTRY, {})
        self.assertEqual(list(retrieval._BM25_CACHE), [("local", "test-version", retrieval.TOKENIZER_VERSION)])

    def test_real_graph_passes_provider_to_every_model_node(self):
        from langgraph_flow.graph_builder import build_graph
        client = self.client()
        client.post("/api/upload", headers=self.headers(), files={"files": ("old.txt", b"alpha original")})
        client.post("/api/provider", headers=self.headers(), json={"provider": "gemini"})
        self.server._graph = build_graph()
        with patch("agents.intent_classifier.invoke_orchestrator", return_value=SimpleNamespace(content="document_qa")) as intent, \
                patch("agents.response_planner.invoke_orchestrator", return_value=SimpleNamespace(content="intent: document_qa")) as planner, \
                patch("agents.qa_agent.invoke_orchestrator", return_value=SimpleNamespace(content="answer")) as qa, \
                patch("retrieval.reranker.get_reranker_model", return_value=None):
            result = client.post("/api/chat", headers=self.headers(), json={"query": "alpha"})
        self.assertEqual(result.status_code, 200)
        intent.assert_not_called()
        for model in (planner, qa):
            self.assertEqual(model.call_args.kwargs["provider"], "gemini")
