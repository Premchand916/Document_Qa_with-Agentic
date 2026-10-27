from unittest.mock import patch

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

