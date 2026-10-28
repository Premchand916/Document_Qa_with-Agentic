"""Step 11 copy-only migration and rollback tests."""
from pathlib import Path

from tests.support import OfflineCase


class MigrationRehearsalTests(OfflineCase):
    def publish(self, repository, text: str, expected: str | None) -> str:
        from langchain_core.documents import Document
        vectorstore = self.store.build_vector_store([
            Document(page_content=text, metadata={"source": "policy.txt"})
        ])
        return repository.publish("alpha", vectorstore, [], [], ["txt"],
                                  [{"name": "policy.txt", "sha256": "a" * 64}],
                                  expected_version=expected)

    def test_copy_rolls_back_restores_restarts_and_never_mutates_source(self):
        from storage.migration_rehearsal import rehearse_rollback
        from storage.workspaces import WorkspaceRepository
        source = self.temp / "source"
        copy = self.temp / "copy"
        repository = WorkspaceRepository(source, lambda: self.embeddings, "ollama")
        old = self.publish(repository, "alpha old policy", None)
        current = self.publish(repository, "alpha current policy", old)
        versions = repository.list_versions("alpha")
        self.assertEqual([item["version"] for item in versions], [current, old])
        self.assertTrue(versions[0]["active"])

        report = rehearse_rollback(source, "alpha", lambda: self.embeddings, copy_root=copy)
        self.assertTrue(report["passed"])
        self.assertFalse(report["source_mutated"])
        self.assertEqual(report["current_version"], current)
        self.assertEqual(report["previous_version"], old)
        self.assertEqual(WorkspaceRepository(source, lambda: self.embeddings).snapshot("alpha").version, current)
        restored = WorkspaceRepository(copy, lambda: self.embeddings).snapshot("alpha")
        self.assertEqual(restored.version, current)
        self.assertIn("current", restored.vectorstore.similarity_search("alpha", k=1)[0].page_content)

    def test_rehearsal_requires_history_and_empty_copy_directory(self):
        from storage.migration_rehearsal import rehearse_rollback
        from storage.workspaces import WorkspaceRepository
        source = self.temp / "source"
        repository = WorkspaceRepository(source, lambda: self.embeddings, "ollama")
        self.publish(repository, "only version", None)
        with self.assertRaisesRegex(ValueError, "previous"):
            rehearse_rollback(source, "alpha", lambda: self.embeddings,
                              copy_root=self.temp / "copy")

        current = repository.state("alpha")["active_version"]
        self.publish(repository, "second version", current)
        occupied = self.temp / "occupied"
        occupied.mkdir()
        (occupied / "keep.txt").write_text("do not overwrite")
        with self.assertRaisesRegex(ValueError, "empty"):
            rehearse_rollback(source, "alpha", lambda: self.embeddings, copy_root=occupied)
        self.assertEqual((occupied / "keep.txt").read_text(), "do not overwrite")
        with self.assertRaisesRegex(ValueError, "outside source"):
            rehearse_rollback(source, "alpha", lambda: self.embeddings,
                              copy_root=source / "unsafe-copy")


if __name__ == "__main__":
    import unittest
    unittest.main()
