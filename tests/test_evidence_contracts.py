"""End-to-end graph evidence and failure contracts."""

from types import SimpleNamespace
from unittest.mock import patch

from tests.support import OfflineCase


class EvidenceContractTests(OfflineCase):
    def test_legacy_unscoped_memory_is_retired(self):
        from memory.conversation_memory import get_memory
        with self.assertRaisesRegex(RuntimeError, "ConversationRepository"):
            get_memory()

    def test_response_plan_reaches_qa_and_citation_resolves(self):
        from langchain_core.documents import Document
        from agents.qa_agent import qa_agent
        from agents.evidence import finalize_answer
        document = Document(page_content="Alpha returns allow 30 days.", metadata={
            "source": "policy.txt", "chunk_id": "stable-alpha", "page": 1,
        })
        state = {"query": "Summarize", "documents": [document], "corpus_version": "version-1",
                 "response_plan": {"sections": ["AUDIT_REQUIRED_SECTION"], "answer_mode": "Executive Brief"}}
        with patch("agents.qa_agent.invoke_orchestrator", return_value=SimpleNamespace(content="30 days")) as model:
            result = finalize_answer(qa_agent(state))
        prompt = model.call_args.args[0]
        self.assertIn("AUDIT_REQUIRED_SECTION", prompt)
        self.assertIn(document.metadata["evidence_id"], prompt)
        self.assertLess(prompt.index("Structured task prompt"), prompt.index("Document Context"))
        self.assertIn("Never repeat conflicting or irrelevant semantic memory", prompt)
        self.assertIn(f'[{document.metadata["evidence_id"]}]', result["response"])
        self.assertEqual(result["citations"][0]["corpus_version"], "version-1")
        self.assertEqual(result["result_status"], "answered")

    def test_unknown_model_citation_fails_closed(self):
        from langchain_core.documents import Document
        from agents.evidence import finalize_answer
        state = {"response": "Unsupported claim [E-ffffffffffff]", "documents": [
            Document(page_content="known", metadata={"source": "known.txt", "chunk_id": "known"})
        ]}
        result = finalize_answer(state)
        self.assertEqual(result["result_status"], "citation_error")
        self.assertEqual(result["citations"], [])
        self.assertNotIn("Unsupported claim", result["response"])

    def test_conflicting_values_keep_both_sources_visible(self):
        from langchain_core.documents import Document
        from agents.evidence import finalize_answer
        docs = [
            Document(page_content="Return deadline is 30 days.", metadata={"source": "old.txt", "chunk_id": "old"}),
            Document(page_content="Return deadline is 60 days.", metadata={"source": "new.txt", "chunk_id": "new"}),
        ]
        result = finalize_answer({"response": "Sources disagree.", "documents": docs})
        self.assertEqual(result["evidence_status"]["status"], "possible_conflict")
        self.assertEqual(result["evidence_status"]["sources"], ["new.txt", "old.txt"])
        self.assertEqual(len(result["documents"]), 2)
        self.assertEqual({item["source"] for item in result["citations"]}, {"old.txt", "new.txt"})

    def test_graph_uses_one_planning_call_and_handles_missing_vectorstore(self):
        from langgraph_flow.graph_builder import build_graph
        with patch("agents.response_planner.invoke_orchestrator",
                   return_value=SimpleNamespace(content="intent: document_qa")) as planner, \
                patch("agents.qa_agent.invoke_orchestrator") as qa:
            result = build_graph().invoke({"query": "What is policy?", "vectorstore": None,
                                           "tabular_assets": [], "use_web_search": False})
        self.assertEqual(planner.call_count, 1)
        qa.assert_not_called()
        self.assertEqual(result["result_status"], "insufficient_evidence")

    def test_table_path_preserves_exact_result_and_adds_local_citation(self):
        import pandas as pd
        from langgraph_flow.graph_builder import build_graph
        asset = {"source": "sales.csv", "sheet": "Data", "file_type": "csv",
                 "source_id": "sales-source", "dataframe": pd.DataFrame({"amount": [10, 20]})}
        with patch("agents.response_planner.invoke_orchestrator",
                   return_value=SimpleNamespace(content="intent: data_analysis")):
            result = build_graph().invoke({"query": "Total amount", "vectorstore": None,
                                           "tabular_assets": [asset], "corpus_version": "table-version"})
        self.assertEqual(result["tool_result"]["value"], 30)
        self.assertEqual(result["result_status"], "answered")
        self.assertEqual(result["citations"][0]["source"], "sales.csv")
        self.assertEqual(result["citations"][0]["corpus_version"], "table-version")

    def test_web_path_labels_origin_and_validates_citation(self):
        from langchain_core.documents import Document
        from langgraph_flow.graph_builder import build_graph
        web = Document(page_content="Current policy is 30 days.", metadata={
            "source": "https://example.invalid/policy", "url": "https://example.invalid/policy",
            "content_type": "web_result", "page": "Web",
        })
        with patch("langgraph_flow.graph_builder.web_search_agent",
                   side_effect=lambda state: {**state, "documents": [web], "source": "Web"}), \
                patch("agents.qa_agent.invoke_orchestrator", return_value=SimpleNamespace(content="30 days")):
            result = build_graph().invoke({"query": "Current policy?", "use_web_search": True})
        self.assertEqual(result["citations"][0]["origin"], "web")
        self.assertIsNone(result["citations"][0]["corpus_version"])
