"""Synthetic inputs and isolated application state for offline tests."""

from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


FIXTURES = Path(__file__).parent / "fixtures"


class Upload(BytesIO):
    def __init__(self, name: str, content: bytes):
        super().__init__(content)
        self.name = name


def text_pdf(text: str) -> bytes:
    """Build a valid single-page ASCII PDF, including byte offsets, without a SDK."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 50 750 Td ({escaped}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
    ]
    result = b"%PDF-1.4\n"
    offsets = [0]
    for index, content in enumerate(objects, 1):
        offsets.append(len(result))
        result += f"{index} 0 obj\n".encode() + content + b"\nendobj\n"
    xref = len(result)
    result += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    result += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    result += f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return result


class OfflineCase(unittest.TestCase):
    """Block external I/O and real models, isolate globals, use temporary indexes."""

    def setUp(self) -> None:
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.network_attempts = []

        def deny_network(*args, **kwargs):
            self.network_attempts.append("blocked")
            raise AssertionError("Network access is forbidden in baseline tests")

        for target in ("socket.socket.connect", "socket.socket.connect_ex",
                       "socket.create_connection", "socket.getaddrinfo", "socket.socket.sendto"):
            self.stack.enter_context(patch(target, side_effect=deny_network))
        # Process safety flags are set once in tests/__init__.py. Restoring the
        # entire environment per test can erase native OpenMP registration keys.
        self.stack.enter_context(patch("dotenv.load_dotenv", return_value=False))

        from langchain_core.embeddings import Embeddings
        import app.server as server
        import agents.retriever_agent as retrieval
        import embeddings.Hugging_face_embedding as embeddings
        import retrieval.reranker as reranker
        import vector_Store.faiss_store as store

        class KeywordEmbeddings(Embeddings):
            def embed_query(self, text: str) -> list[float]:
                return [1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0]

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [self.embed_query(text) for text in texts]

        self.embeddings = KeywordEmbeddings()
        self.server = server
        self.store = store
        self.temp = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(store, "DB_PATH", self.temp / "vector_db"))
        self.stack.enter_context(patch.object(store, "get_cached_embedding", return_value=self.embeddings))
        from storage.workspaces import WorkspaceRepository
        self.stack.enter_context(patch.object(server, "get_repository", side_effect=lambda: WorkspaceRepository(self.temp / "vector_db", lambda: self.embeddings, "ollama")))
        self.stack.enter_context(patch.object(server, "parse_in_isolated_process",
                                      side_effect=lambda name, content: server.parse_uploaded_file(Upload(name, content))))
        self.stack.enter_context(patch.object(server, "_graph", None))
        self.stack.enter_context(patch("utils.prompt_skill.get_prompt_skill", return_value={"body": ""}))
        self.stack.enter_context(patch.object(embeddings, "HuggingFaceEmbeddings", side_effect=AssertionError("Real embeddings forbidden")))
        self.stack.enter_context(patch.object(reranker, "CrossEncoder", side_effect=AssertionError("Real reranker forbidden")))
        for module in ("intent_classifier", "response_planner", "qa_agent", "question_generator", "orchestrator_agent"):
            self.stack.enter_context(patch(f"agents.{module}.invoke_orchestrator", side_effect=AssertionError("Real LLM forbidden")))
        retrieval.clear_retrieval_caches()
        embeddings.get_embedding_model.cache_clear()
        reranker.get_reranker_model.cache_clear()
        self.addCleanup(retrieval.clear_retrieval_caches)
        self.addCleanup(embeddings.get_embedding_model.cache_clear)
        self.addCleanup(reranker.get_reranker_model.cache_clear)

    def tearDown(self) -> None:
        self.assertEqual(self.network_attempts, [], "A test attempted network I/O, even if the app swallowed it")

    def seed_workspace(self, documents=None, workspace="local"):
        from langchain_core.documents import Document
        documents = documents or [Document(page_content="alpha policy", metadata={"source": "policy.txt"})]
        repository = self.server.get_repository()
        previous = repository.state(workspace)["active_version"]
        vectorstore = self.store.build_vector_store(documents)
        repository.publish(workspace, vectorstore, [], [], ["txt"], [], expected_version=previous)
        return vectorstore

    def client(self):
        from fastapi.testclient import TestClient
        # No lifespan: most tests supply state; one explicitly tests startup.
        client = TestClient(self.server.app, raise_server_exceptions=False, client=("127.0.0.1", 12345), base_url="http://127.0.0.1")
        self.addCleanup(client.close)
        return client

    def stub_graph_models(self) -> None:
        for module, output in (("intent_classifier", "document_qa"),
                               ("response_planner", "intent: document_qa"),
                               ("qa_agent", "The alpha deadline is 30 days.")):
            self.stack.enter_context(patch(f"agents.{module}.invoke_orchestrator", return_value=SimpleNamespace(content=output)))

