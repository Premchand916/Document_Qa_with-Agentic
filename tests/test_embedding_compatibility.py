"""A corpus must never be searched with a different embedding coordinate system."""

import os
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from tests.support import OfflineCase


class Axes(Embeddings):
    def __init__(self, dimension=2, reverse=False):
        self.dimension = dimension
        self.reverse = reverse

    def embed_query(self, text: str) -> list[float]:
        first = ("alpha" in text.lower()) != self.reverse
        return ([1.0, 0.0] if first else [0.0, 1.0]) + [0.0] * (self.dimension - 2)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class EmbeddingCompatibilityTests(OfflineCase):
    def corpus(self, embedding):
        from langchain_community.vectorstores import FAISS
        return FAISS.from_documents([Document(page_content="alpha policy", metadata={"source": "alpha.txt"}),
                                    Document(page_content="beta policy", metadata={"source": "beta.txt"})], embedding)

    def repository(self, embedding):
        from storage.workspaces import WorkspaceRepository
        return WorkspaceRepository(self.temp / "compatible", lambda: embedding)

    def publish(self, embedding):
        repo = self.repository(embedding)
        repo.publish("alpha", self.corpus(embedding), [], [], ["txt"], [], expected_version=None)

    def test_dimension_change_is_rejected_before_native_index_read(self):
        from storage.workspaces import CorpusIntegrityError
        self.publish(Axes())
        with patch("faiss.read_index") as native_reader:
            with self.assertRaises(CorpusIntegrityError):
                self.repository(Axes(dimension=3)).snapshot("alpha")
        native_reader.assert_not_called()

    def test_same_dimension_incompatible_encoder_is_rejected(self):
        from storage.workspaces import CorpusIntegrityError
        self.publish(Axes())
        with self.assertRaises(CorpusIntegrityError):
            self.repository(Axes(reverse=True)).snapshot("alpha")
        restored = self.repository(Axes()).snapshot("alpha")
        self.assertEqual(restored.vectorstore.similarity_search("alpha", k=1)[0].metadata["source"], "alpha.txt")

    def test_publish_rejects_index_built_with_other_encoder(self):
        from storage.workspaces import CorpusIntegrityError
        with self.assertRaises(CorpusIntegrityError):
            self.repository(Axes(reverse=True)).publish("alpha", self.corpus(Axes()), [], [], ["txt"], [], expected_version=None)
        self.assertIsNone(self.repository(Axes()).state("alpha")["active_version"])

    def test_embedding_model_cache_follows_configured_name(self):
        import embeddings.Hugging_face_embedding as module
        module.get_embedding_model.cache_clear()
        with patch.object(module, "HuggingFaceEmbeddings", side_effect=lambda **kwargs: kwargs["model_name"]):
            with patch.dict(os.environ, {"EMBEDDING_MODEL": "first"}):
                first = module.get_embedding_model()
            with patch.dict(os.environ, {"EMBEDDING_MODEL": "second"}):
                second = module.get_embedding_model()
        self.assertEqual((first, second), ("first", "second"))

    def test_embedding_loader_uses_standard_cache_when_project_cache_is_empty(self):
        import embeddings.Hugging_face_embedding as module
        module.get_embedding_model.cache_clear()
        with patch.object(module, "_cached_model_exists", return_value=False), \
             patch.object(module, "HuggingFaceEmbeddings", side_effect=lambda **kwargs: kwargs) as loader:
            result = module._load_embedding_model("cached-model", True, "")
        self.assertIsNone(result["cache_folder"])
        self.assertEqual(result["model_kwargs"], {"local_files_only": True})
        self.assertEqual(result["encode_kwargs"], {"batch_size": 32})
        self.assertEqual(loader.call_count, 1)
        module.get_embedding_model.cache_clear()
        with patch.object(module, "_cached_model_exists", return_value=True), \
             patch.object(module, "HuggingFaceEmbeddings", side_effect=lambda **kwargs: kwargs):
            project_cached = module._load_embedding_model("cached-model", True, "")
        self.assertEqual(project_cached["cache_folder"], str(module.MODEL_CACHE_DIR))
        module.get_embedding_model.cache_clear()

    def test_embedding_batch_size_is_bounded_and_selects_cache_entry(self):
        import embeddings.Hugging_face_embedding as module
        module.get_embedding_model.cache_clear()
        with patch.object(module, "HuggingFaceEmbeddings", side_effect=lambda **kwargs: kwargs):
            with patch.dict(os.environ, {"EMBEDDING_MODEL": "cached-model", "EMBEDDING_BATCH_SIZE": "8"}):
                first = module.get_embedding_model()
            with patch.dict(os.environ, {"EMBEDDING_MODEL": "cached-model", "EMBEDDING_BATCH_SIZE": "16"}):
                second = module.get_embedding_model()
        self.assertEqual(first["encode_kwargs"]["batch_size"], 8)
        self.assertEqual(second["encode_kwargs"]["batch_size"], 16)
        for value in ("0", "257", "invalid"):
            with patch.dict(os.environ, {"EMBEDDING_BATCH_SIZE": value}):
                with self.assertRaisesRegex(ValueError, "EMBEDDING_BATCH_SIZE"):
                    module.get_embedding_model()
        module.get_embedding_model.cache_clear()

    def test_loaded_weights_and_tokenizer_both_affect_fingerprint(self):
        import torch
        from embeddings.identity import embedding_identity

        class Tokenizer:
            name_or_path = "synthetic"
            special_tokens_map = {"cls_token": "[CLS]"}
            vocab_size = 2

            def __init__(self, token_id):
                self.token_id = token_id

            def get_vocab(self):
                return {"alpha": self.token_id, "beta": 1}

        class Client:
            _modules = {}

            def __init__(self, weight, token_id):
                self.weight = weight
                self.tokenizer = Tokenizer(token_id)

            def state_dict(self):
                return {"weights": torch.tensor([self.weight], dtype=torch.float32)}

            def get_sentence_embedding_dimension(self):
                return 2

        def identity(weight, token_id, batch_size=32):
            return embedding_identity(SimpleNamespace(client=Client(weight, token_id), model_name="synthetic",
                                                      model_kwargs={}, encode_kwargs={"batch_size": batch_size},
                                                      query_encode_kwargs={}))

        self.assertNotEqual(identity(1, 1), identity(2, 1))
        self.assertNotEqual(identity(1, 1), identity(1, 3))
        self.assertEqual(identity(1, 1, 8), identity(1, 1, 64))

    def test_rollback_activates_only_a_compatible_saved_version(self):
        from storage.workspaces import CorpusIntegrityError, PublicationConflict
        first_model, second_model = Axes(), Axes(reverse=True)
        first = self.repository(first_model).publish("alpha", self.corpus(first_model), [], [], ["txt"], [], expected_version=None)
        second = self.repository(second_model).publish("alpha", self.corpus(second_model), [], [], ["txt"], [], expected_version=first)
        with self.assertRaises(CorpusIntegrityError):
            self.repository(second_model).activate_version("alpha", first, expected_version=second)
        self.assertEqual(self.repository(second_model).state("alpha")["active_version"], second)
        restored = self.repository(first_model)
        restored.activate_version("alpha", first, expected_version=second)
        self.assertEqual(restored.snapshot("alpha").version, first)
        self.assertEqual(restored.snapshot("alpha").vectorstore.similarity_search("alpha", k=1)[0].metadata["source"], "alpha.txt")
        with self.assertRaises(PublicationConflict):
            restored.activate_version("alpha", first, expected_version=second)
        self.assertEqual(restored.state("alpha")["active_version"], first)
