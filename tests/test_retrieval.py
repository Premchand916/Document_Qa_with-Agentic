from types import SimpleNamespace
from unittest.mock import Mock, patch

from tests.support import OfflineCase


class RetrievalTests(OfflineCase):
    def documents(self):
        from langchain_core.documents import Document
        return [Document(page_content="alpha deadline is 30 days", metadata={"source": "alpha.txt"}),
                Document(page_content="beta deadline is 60 days", metadata={"source": "beta.txt"})]

    def test_faiss_roundtrip_preserves_vectors_text_and_source(self):
        self.store.create_vector_store(self.documents())
        loaded = self.store.load_vector_store_direct()
        self.assertEqual(loaded.index.ntotal, 2)
        found = loaded.similarity_search("alpha deadline", k=1)[0]
        self.assertEqual(found.metadata["source"], "alpha.txt")
        self.assertEqual(found.page_content, "alpha deadline is 30 days")

    def test_missing_index_returns_none(self):
        self.assertIsNone(self.store.load_vector_store_direct())

    def test_reranking_uses_query_document_pairs_and_descending_scores(self):
        from retrieval.reranker import rerank_documents
        from unittest.mock import Mock
        model = Mock()
        model.predict.return_value = [0.1, 0.9]
        docs = self.documents()
        with patch("retrieval.reranker.get_reranker_model", return_value=model):
            result = rerank_documents("beta deadline", docs, top_k=1)
        self.assertEqual(result, [docs[1]])
        model.predict.assert_called_once_with([("beta deadline", d.page_content) for d in docs])

    def test_empty_candidates_do_not_load_reranker(self):
        from retrieval.reranker import rerank_documents
        with patch("retrieval.reranker.get_reranker_model") as loader:
            self.assertEqual(rerank_documents("alpha", []), [])
        loader.assert_not_called()

    def test_reranker_failure_keeps_candidate_order(self):
        from retrieval.reranker import rerank_documents
        docs = self.documents()
        with patch("retrieval.reranker.get_reranker_model", side_effect=RuntimeError("unavailable")):
            self.assertEqual(rerank_documents("alpha", docs, top_k=1), docs[:1])

    def test_fusion_rewards_agreement_without_losing_distinct_text(self):
        from agents.retriever_agent import _rrf_fusion
        alpha, beta = self.documents()
        result = _rrf_fusion([[alpha, beta], [beta]])
        self.assertEqual(result, [beta, alpha])

    def test_identical_text_keeps_distinct_source_and_chunk_provenance(self):
        from langchain_core.documents import Document
        from agents.retriever_agent import _rrf_fusion
        first = Document(page_content="same", metadata={"source": "a", "chunk_id": "a:1"})
        second = Document(page_content="same", metadata={"source": "b", "chunk_id": "b:1"})
        copy_of_first = Document(page_content="same", metadata={"source": "a", "chunk_id": "a:1"})
        fused = _rrf_fusion([[first, second], [copy_of_first]])
        self.assertEqual([doc.metadata["source"] for doc in fused], ["a", "b"])

    def test_lexical_normalization_preserves_identifiers_and_discards_zero_overlap(self):
        from agents.retriever_agent import _tokens, retriever_agent
        self.assertEqual(_tokens("REFUND? INV-2048 v2.1"), ["refund", "inv-2048", "v2.1"])
        docs = self.documents()
        vs = SimpleNamespace(docstore=SimpleNamespace(_dict=dict(enumerate(docs))),
                             similarity_search=lambda query, k: [])
        absent = retriever_agent({"query": "xylophone warranty", "vectorstore": vs})
        self.assertEqual(absent["documents"], [])
        self.assertEqual(absent["retrieval_status"]["lexical_hits"], 0)
        with patch("agents.retriever_agent.rerank_with_status") as rerank:
            from retrieval.reranker import RerankResult
            rerank.side_effect = lambda query, found, top_k: RerankResult(found[:top_k], None, None, "fallback")
            matched = retriever_agent({"query": "ALPHA deadline?", "vectorstore": vs})
        self.assertEqual([doc.metadata["source"] for doc in matched["documents"]], ["alpha.txt", "beta.txt"])

    def test_reranker_recovers_after_backoff_and_rejects_malformed_scores(self):
        import retrieval.reranker as reranker
        reranker.get_reranker_model.cache_clear()
        model = Mock()
        model.predict.return_value = [float("nan"), 0.9]
        with patch.object(reranker.time, "monotonic", side_effect=[0.0, 1.0, 6.0]), \
                patch.object(reranker, "CrossEncoder", side_effect=[RuntimeError("temporary"), model]) as loader:
            self.assertIsNone(reranker.get_reranker_model())
            self.assertIsNone(reranker.get_reranker_model())
            self.assertIs(reranker.get_reranker_model(), model)
        self.assertEqual(loader.call_count, 2)
        docs = self.documents()
        result = reranker.rerank_with_status("deadline", docs)
        self.assertEqual(result.documents, docs)
        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.degradation_reason, "invalid_scores_or_inference")
        model.predict.return_value = [0.2]
        self.assertEqual(reranker.rerank_with_status("deadline", docs).status, "fallback")
        model.predict.return_value = [0.2, 0.9]
        ready = reranker.rerank_with_status("deadline", docs)
        self.assertEqual(ready.documents, [docs[1], docs[0]])
        self.assertEqual(ready.scores, [0.9, 0.2])

    def test_workspace_bm25_cache_is_bounded_and_versioned(self):
        from agents import retriever_agent as retrieval
        docs = tuple(self.documents())
        for i in range(retrieval.MAX_BM25_INDEXES + 1):
            retrieval._workspace_bm25({"request_context": {"workspace_id": "alpha"},
                                       "corpus_version": f"v{i}"}, docs)
        self.assertEqual(len(retrieval._BM25_CACHE), retrieval.MAX_BM25_INDEXES)
        self.assertNotIn(("alpha", "v0", retrieval.TOKENIZER_VERSION), retrieval._BM25_CACHE)

    def test_absent_evidence_returns_explicit_insufficient_status(self):
        from agents.qa_agent import qa_agent
        state = qa_agent({"query": "xylophone warranty", "documents": []})
        self.assertEqual(state["result_status"], "insufficient_evidence")
        self.assertIn("No relevant information", state["response"])

    def test_enabled_reranker_and_fallback_use_the_same_candidate_set(self):
        from retrieval.reranker import rerank_with_status
        docs = self.documents()
        model = Mock()
        model.predict.return_value = [0.1, 0.9]
        with patch("retrieval.reranker.get_reranker_model", return_value=model):
            enabled = rerank_with_status("beta deadline", docs, top_k=2)
        with patch("retrieval.reranker.get_reranker_model", return_value=None):
            fallback = rerank_with_status("beta deadline", docs, top_k=2)
        self.assertEqual(enabled.documents, [docs[1], docs[0]])
        self.assertEqual(fallback.documents, docs)
        self.assertEqual(enabled.status, "ready")
        self.assertEqual(fallback.status, "fallback")

