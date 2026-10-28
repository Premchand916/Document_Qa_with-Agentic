import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from tests.support import FIXTURES, OfflineCase, text_pdf


class ApiAndGraphTests(OfflineCase):
    def test_conversation_ui_exposes_delete_pagination_and_mobile_layout(self):
        from app.server import STATIC_DIR
        html = (STATIC_DIR / "index.html").read_text()
        self.assertIn("deleteConversation()", html)
        self.assertIn("loadOlderConversation()", html)
        self.assertIn("submitCorrection()", html)
        self.assertIn("waitForIngestion(jobId)", html)
        self.assertIn("cancelIngestion()", html)
        self.assertIn("@media (max-width:760px)", html)
        self.assertIn("Delete this conversation permanently?", html)

    def test_chat_before_startup_returns_service_unavailable(self):
        self.assertEqual(self.client().post("/api/chat", json={"query": "alpha"}).status_code, 503)

    def test_missing_query_returns_validation_error(self):
        self.assertEqual(self.client().post("/api/chat", json={}).status_code, 422)

    def test_no_corpus_returns_actionable_error(self):
        self.server._graph = SimpleNamespace(invoke=lambda state: self.fail("Must not run"))
        response = self.client().post("/api/chat", json={"query": "alpha"})
        self.assertEqual(response.status_code, 422)
        self.assertIn("Upload", response.json()["detail"])

    def test_invalid_provider_does_not_change_active_provider(self):
        client = self.client()
        before = client.get("/api/state").json()["provider"]
        self.assertEqual(client.post("/api/provider", json={"provider": "invalid"}).status_code, 422)
        self.assertEqual(client.get("/api/state").json()["provider"], before)

    def test_unsupported_upload_is_rejected_without_creating_index(self):
        response = self.client().post("/api/upload", files={"files": ("file.exe", b"synthetic")})
        self.assertEqual(response.status_code, 422)
        self.assertFalse((self.store.DB_PATH / "corpora").exists())
        self.assertIsNone(self.server.get_repository().state("local")["active_version"])

    def test_upload_to_document_answer_with_real_graph_and_faiss(self):
        from langgraph_flow.graph_builder import build_graph
        self.stub_graph_models()
        self.server._graph = build_graph()
        self.stack.enter_context(patch.object(self.server, "generate_questions", return_value=[]))
        self.stack.enter_context(patch("retrieval.reranker.get_reranker_model", return_value=None))
        client = self.client()
        upload = client.post("/api/upload", files={"files": ("policy.txt", (FIXTURES / "corpus" / "policy.txt").read_bytes())})
        self.assertEqual(upload.status_code, 200, upload.text)
        self.assertTrue(upload.json()["vectorstore_ready"])
        response = client.post("/api/chat", json={"query": "What is the alpha deadline?"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("30 days", response.json()["response"])
        self.assertEqual(response.json()["documents"][0]["metadata"]["source"], "policy.txt")
        self.assertEqual(response.json()["retrieval_status"]["reranker"]["status"], "fallback")
        self.assertEqual(len(response.json()["citations"]), 1)
        self.assertEqual(response.json()["citations"][0]["corpus_version"], response.json()["corpus_version"])

    def test_upload_to_table_total_with_real_graph(self):
        from langgraph_flow.graph_builder import build_graph
        self.stub_graph_models()
        self.server._graph = build_graph()
        self.stack.enter_context(patch.object(self.server, "generate_questions", return_value=[]))
        client = self.client()
        upload = client.post("/api/upload", files={"files": ("sales.csv", (FIXTURES / "corpus" / "sales.csv").read_bytes())})
        self.assertEqual(upload.status_code, 200, upload.text)
        response = client.post("/api/chat", json={"query": "Total amount", "uploaded_file_types": ["CSV"]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("amount: 30", response.json()["response"])
        self.assertIn("sales.csv", response.json()["source"])

    def test_multi_table_api_selects_second_filters_north_and_clarifies_ambiguity(self):
        from langgraph_flow.graph_builder import build_graph
        self.stub_graph_models()
        self.server._graph = build_graph()
        self.stack.enter_context(patch.object(self.server, "generate_questions", return_value=[]))
        client = self.client()
        files = [("files", ("sales.csv", b"region,amount\nNorth,10\nSouth,20\n")),
                 ("files", ("second.csv", b"region,amount\nWest,99\n"))]
        self.assertEqual(client.post("/api/upload", files=files).status_code, 200)
        second = client.post("/api/chat", json={"query": "Total amount in second.csv"}).json()
        self.assertEqual(second["tool_result"]["value"], 99)
        self.assertEqual(second["source"], "second.csv (Data)")
        north = client.post("/api/chat", json={"query": "Total amount for North only in sales.csv"}).json()
        self.assertEqual(north["tool_result"]["value"], 10)
        self.assertEqual(north["tool_result"]["selected_rows"], [1])
        ambiguous = client.post("/api/chat", json={"query": "Total amount"}).json()
        self.assertEqual(ambiguous["result_status"], "needs_clarification")
        self.assertIsNone(ambiguous["tool_result"])

    def test_pdf_question_uses_retrieval_after_mixed_upload(self):
        from langgraph_flow.graph_builder import build_graph
        self.stub_graph_models()
        self.server._graph = build_graph()
        self.stack.enter_context(patch.object(self.server, "generate_questions", return_value=[]))
        self.stack.enter_context(patch("retrieval.reranker.get_reranker_model", return_value=None))
        client = self.client()
        files = [("files", ("policy.pdf", text_pdf("Alpha cancellation policy 30 days"))),
                 ("files", ("sales.csv", b"region,amount\nNorth,10\n"))]
        upload = client.post("/api/upload", files=files)
        self.assertEqual(upload.status_code, 200)
        self.assertEqual(upload.json()["extraction_reports"][0]["pages_with_coordinates"], [1])
        self.assertEqual(client.get("/api/state").json()["extraction_reports"], upload.json()["extraction_reports"])
        saved = list(self.server.get_repository().snapshot("local").vectorstore.docstore._dict.values())
        self.assertTrue(any(doc.metadata.get("file_type") == "pdf" and doc.metadata.get("bbox") for doc in saved))
        answer = client.post("/api/chat", json={"query": "What is the PDF cancellation policy?"}).json()
        self.assertIsNone(answer["tool_result"])
        self.assertTrue(answer["documents"])
        self.assertNotEqual(answer["documents"][0]["metadata"].get("content_type"), "dataframe_result")

    def test_long_json_tail_and_chunk_provenance_survive_publication(self):
        self.stack.enter_context(patch.object(self.server, "generate_questions", return_value=[]))
        payload = ('{"body":"' + 'alpha ' * 1000 + 'FINAL_FACT_42"}').encode()
        response = self.client().post("/api/upload", files={"files": ("long.json", payload)})
        self.assertEqual(response.status_code, 200, response.text)
        snapshot = self.server.get_repository().snapshot("local")
        documents = list(snapshot.vectorstore.docstore._dict.values())
        self.assertTrue(any("FINAL_FACT_42" in doc.page_content for doc in documents))
        self.assertTrue(all(doc.metadata["source_id"] and doc.metadata["chunk_id"] for doc in documents))
        self.assertTrue(all(doc.metadata["json_pointer"] == "/body" for doc in documents))
        self.assertEqual(len({doc.metadata["chunk_id"] for doc in documents}), len(documents))

    def test_upload_warning_remains_visible_after_state_reload(self):
        from langchain_core.documents import Document
        from ingestion.file_loader import ParsedFile
        self.stack.enter_context(patch.object(self.server, "generate_questions", return_value=[]))
        report = {"source": "partial.pdf", "file_type": "pdf", "text_extracted": True,
                  "ocr_engine": None, "page_count": 2, "text_pages": 1,
                  "pages_without_text": [2], "pages_with_coordinates": [1]}
        parsed = ParsedFile([Document(page_content="alpha text", metadata={"source": "partial.pdf", "page": 1})],
                            [], ["partial.pdf: page 2 needs OCR"], report)
        with patch.object(self.server, "parse_uploaded_file", return_value=parsed):
            upload = self.client().post("/api/upload", files={"files": ("partial.pdf", b"synthetic")})
        self.assertEqual(upload.status_code, 200, upload.text)
        self.assertEqual(upload.json()["warnings"], parsed.warnings)
        self.assertEqual(self.client().get("/api/state").json()["warnings"], parsed.warnings)
        self.assertEqual(upload.json()["extraction_reports"], [report])
        self.assertEqual(self.client().get("/api/state").json()["extraction_reports"], [report])

    def test_scanned_pdf_ocr_publishes_searchable_corpus_and_report(self):
        from io import BytesIO
        from PIL import Image, ImageDraw
        import ingestion.file_loader as loader
        image = Image.new("RGB", (900, 260), "white")
        ImageDraw.Draw(image).text((40, 100), "SCANNED POLICY", fill="black")
        output = BytesIO()
        image.save(output, format="PDF", resolution=150)
        with patch.object(loader, "_tesseract_text", return_value="ALPHA OCR DEADLINE 30 DAYS"), \
                patch.object(self.server, "generate_questions", return_value=[]):
            upload = self.client().post("/api/upload", files={"files": ("scan.pdf", output.getvalue())})
        self.assertEqual(upload.status_code, 200, upload.text)
        self.assertEqual(upload.json()["extraction_reports"][0]["ocr_engine"], "tesseract")
        self.assertEqual(upload.json()["extraction_reports"][0]["ocr_pages"], [1])
        snapshot = self.server.get_repository().snapshot("local")
        documents = list(snapshot.vectorstore.docstore._dict.values())
        self.assertTrue(any("ALPHA OCR DEADLINE 30 DAYS" in doc.page_content for doc in documents))

    def test_startup_restores_existing_document_index(self):
        from langchain_core.documents import Document
        self.seed_workspace([Document(page_content="alpha", metadata={"source": "policy.txt"})])
        asyncio.run(self.server._startup())
        self.assertIsNotNone(self.server._graph)
        self.assertEqual(self.server.get_repository().snapshot("local").vectorstore.index.ntotal, 1)

    def test_web_route_uses_supplied_search_evidence(self):
        from langchain_core.documents import Document
        from langgraph_flow.graph_builder import build_graph
        self.stub_graph_models()
        evidence = Document(page_content="alpha deadline is 30 days", metadata={"source": "https://example.invalid/policy", "content_type": "web_result"})
        with patch("langgraph_flow.graph_builder.web_search_agent", side_effect=lambda state: {**state, "documents": [evidence]}):
            result = build_graph().invoke({"query": "alpha deadline", "use_web_search": True})
        self.assertIn("30 days", result["response"])
        self.assertEqual(result["documents"], [evidence])

    def test_qa_without_evidence_does_not_call_model(self):
        from agents.qa_agent import qa_agent
        with patch("agents.qa_agent.invoke_orchestrator") as model:
            result = qa_agent({"query": "alpha", "documents": []})
        self.assertIn("No relevant information", result["response"])
        model.assert_not_called()
