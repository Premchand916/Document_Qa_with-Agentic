"""Offline regression coverage for the governed connector foundation."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from connectors.managed_folder import (ConnectorLimitError, ConnectorSourceUnavailable,
                                       ManagedFolderAdapter)
from connectors.service import (ConnectorAccessDenied, ConnectorService,
                                ConnectorSyncError)
from storage.connectors import (ConnectorNotFound, ConnectorRepository,
                                ConnectorRevoked, ConnectorCapacityError)
from tests.support import OfflineCase


class ConnectorFoundationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sources = self.root / "managed"
        (self.sources / "operations").mkdir(parents=True)
        (self.sources / "operations" / "sop.txt").write_text("approved procedure")
        self.repository = ConnectorRepository(self.root / "data")
        self.service = ConnectorService(
            self.repository, ManagedFolderAdapter(self.sources),
            allow_unlisted_principals=frozenset({"alice"}),
        )

    def create_source(self, principal="alice", key="operations"):
        return self.service.create_source(
            "workspace", principal, key, "Operations SOP", "create-source-0001",
        )

    def test_sync_is_idempotent_and_immutable_snapshots_track_changes(self):
        source = self.create_source()
        first = self.service.sync(
            "workspace", "alice", source["source_id"], "sync-source-0001",
        )
        self.assertEqual(first["status"], "ready")
        self.assertTrue(first["changed"])

        duplicate = self.service.sync(
            "workspace", "alice", source["source_id"], "sync-source-0001",
        )
        unchanged = self.service.sync(
            "workspace", "alice", source["source_id"], "sync-source-0002",
        )
        self.assertEqual(duplicate["run_id"], first["run_id"])
        self.assertFalse(unchanged["changed"])
        self.assertEqual(unchanged["snapshot_id"], first["snapshot_id"])

        (self.sources / "operations" / "sop.txt").write_text("revised procedure")
        revised = self.service.sync(
            "workspace", "alice", source["source_id"], "sync-source-0003",
        )
        self.assertNotEqual(revised["snapshot_id"], first["snapshot_id"])

        (self.sources / "operations" / "sop.txt").write_text("approved procedure")
        reverted = self.service.sync(
            "workspace", "alice", source["source_id"], "sync-source-0004",
        )
        self.assertEqual(reverted["snapshot_id"], first["snapshot_id"])
        snapshots = self.service.list_snapshots(
            "workspace", "alice", source["source_id"], limit=10,
        )
        self.assertEqual(len(snapshots), 2)
        original_files = self.repository.load_snapshot_files(
            "workspace", "alice", first["snapshot_id"],
        )
        self.assertEqual(original_files[0][1], b"approved procedure")
        self.assertEqual(original_files[0][2]["connector_source_id"], source["source_id"])

    def test_empty_scan_represents_deletion_and_repository_survives_restart(self):
        source = self.create_source()
        self.service.sync("workspace", "alice", source["source_id"], "sync-delete-0001")
        (self.sources / "operations" / "sop.txt").unlink()
        deleted = self.service.sync(
            "workspace", "alice", source["source_id"], "sync-delete-0002",
        )
        self.assertTrue(deleted["changed"])
        snapshots = ConnectorRepository(self.root / "data").list_snapshots(
            "workspace", "alice", source["source_id"], limit=10,
        )
        deleted_snapshot = next(item for item in snapshots
                                if item["snapshot_id"] == deleted["snapshot_id"])
        self.assertEqual(deleted_snapshot["file_count"], 0)

    def test_owner_scope_acl_freshness_and_revocation_fail_closed(self):
        source = self.create_source()
        with self.assertRaises(ConnectorNotFound):
            self.repository.get_source("workspace", "bob", source["source_id"])
        denied = ConnectorService(
            self.repository, ManagedFolderAdapter(self.sources),
            source_acl={"operations": frozenset({"bob"})},
        )
        with self.assertRaises(ConnectorAccessDenied):
            denied.sync("workspace", "alice", source["source_id"], "sync-denied-0001")

        synced = self.service.sync(
            "workspace", "alice", source["source_id"], "sync-revoke-0001",
        )
        last = self.repository.get_source("workspace", "alice", source["source_id"])["last_synced_at"]
        self.assertEqual(self.service.list_sources("workspace", "alice", now=last)[0]["freshness"], "fresh")
        self.assertEqual(self.service.list_sources(
            "workspace", "alice", now=last + 86401,
        )[0]["freshness"], "stale")

        self.service.revoke_grant("workspace", "alice", source["grant_id"])
        self.assertEqual(self.service.list_sources("workspace", "alice")[0]["freshness"], "revoked")
        with self.assertRaises(ConnectorRevoked):
            self.repository.load_snapshot_files("workspace", "alice", synced["snapshot_id"])
        with self.assertRaises(ConnectorRevoked):
            self.service.sync("workspace", "alice", source["source_id"], "sync-revoke-0002")

    def test_bounded_adapter_and_safe_failed_run(self):
        source = self.create_source()
        (self.sources / "operations" / "unsupported.exe").write_bytes(b"unsafe")
        with self.assertRaises(ConnectorSyncError) as failure:
            self.service.sync("workspace", "alice", source["source_id"], "sync-failure-0001")
        self.assertEqual(failure.exception.code, "source_unavailable")
        repeated = self.service.sync(
            "workspace", "alice", source["source_id"], "sync-failure-0001",
        )
        self.assertEqual(repeated["status"], "failed")
        self.assertEqual(repeated["error_code"], "source_unavailable")

        (self.sources / "operations" / "unsupported.exe").unlink()
        with patch.object(ManagedFolderAdapter, "MAX_FILE_BYTES", 2):
            with self.assertRaises(ConnectorLimitError):
                ManagedFolderAdapter(self.sources).scan("operations")

    def test_durable_source_and_snapshot_counts_are_bounded(self):
        source = self.create_source()
        (self.sources / "other").mkdir()
        with patch.object(ConnectorRepository, "MAX_SOURCES_PER_OWNER", 1):
            with self.assertRaises(ConnectorCapacityError):
                self.service.create_source(
                    "workspace", "alice", "other", "Other", "create-source-0002",
                )

        with patch.object(ConnectorRepository, "MAX_SNAPSHOTS_PER_SOURCE", 1):
            self.service.sync(
                "workspace", "alice", source["source_id"], "sync-capacity-0001",
            )
            (self.sources / "operations" / "sop.txt").write_text("new state")
            with self.assertRaises(ConnectorSyncError) as failure:
                self.service.sync(
                    "workspace", "alice", source["source_id"], "sync-capacity-0002",
                )
        self.assertEqual(failure.exception.code, "snapshot_capacity")

    def test_path_traversal_and_symlinks_are_rejected(self):
        for key in ("../operations", "/operations", "operations/../../private"):
            with self.assertRaises(ValueError):
                ManagedFolderAdapter(self.sources).scan(key)
        target = self.root / "private.txt"
        target.write_text("private")
        link = self.sources / "operations" / "linked.txt"
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest("Symlinks are unavailable on this filesystem")
        with self.assertRaises(ConnectorSourceUnavailable):
            ManagedFolderAdapter(self.sources).scan("operations")
        link.unlink()
        hardlink = self.sources / "operations" / "hardlinked.txt"
        try:
            os.link(target, hardlink)
        except OSError:
            self.skipTest("Hardlinks are unavailable on this filesystem")
        with self.assertRaises(ConnectorSourceUnavailable):
            ManagedFolderAdapter(self.sources).scan("operations")
        hardlink.unlink()
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "nested").mkdir()
        intermediate = self.sources / "alias"
        intermediate.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ConnectorSourceUnavailable):
            ManagedFolderAdapter(self.sources).scan("alias/nested")


class ConnectorApiTests(OfflineCase):
    def test_shared_workspace_connector_is_principal_scoped_and_acl_removal_redacts(self):
        from app.security import AccessSettings
        from fastapi.testclient import TestClient
        from langgraph_flow.graph_builder import build_graph
        self.stub_graph_models()
        self.server._graph = build_graph()
        self.stack.enter_context(patch("retrieval.reranker.get_reranker_model", return_value=None))
        source_root = self.temp / "connector-root"
        (source_root / "policies").mkdir(parents=True)
        (source_root / "policies" / "alpha.txt").write_text("Alpha policy deadline is 30 days.")
        settings = AccessSettings(
            mode="auth_proxy", public_origin="https://docs.example.test",
            workspace_id="local", proxy_secret="synthetic-test-secret-" * 3,
            workspace_memberships={"local": ("alice", "bob")},
        )
        environment = {"DOCQA_CONNECTOR_ROOT": str(source_root),
                       "DOCQA_CONNECTOR_SOURCE_ACL": '{"policies":["alice"]}'}
        with patch.dict(os.environ, environment), \
                patch("app.security.get_access_settings", return_value=settings), \
                TestClient(self.server.app, raise_server_exceptions=False,
                           client=("127.0.0.1", 12345),
                           base_url="https://docs.example.test") as client:
            alice = {"x-docqa-proxy-secret": settings.proxy_secret,
                     "x-auth-request-user": "alice"}
            bob = {**alice, "x-auth-request-user": "bob"}
            source = client.post("/api/connectors/sources", headers=alice, json={
                "source_key": "policies", "display_name": "Policies",
                "idempotency_key": "create-shared-0001",
            }).json()["source"]
            path = f"/api/connectors/sources/{source['source_id']}"
            self.assertEqual(client.post(path+"/sync", headers={**alice,
                "idempotency-key": "sync-shared-0001"}).status_code, 200)
            self.assertEqual(client.post(path+"/publish", headers=alice).status_code, 200)
            self.assertTrue(client.get("/api/state", headers=alice).json()["vectorstore_ready"])
            self.assertFalse(client.get("/api/state", headers=bob).json()["vectorstore_ready"])
            self.assertEqual(client.post(path+"/publish", headers=bob).status_code, 404)
            answer = client.post("/api/chat", headers=alice,
                                 json={"query": "What is the Alpha policy deadline?"}).json()
            self.assertEqual(answer["connector_source_ids"], [source["source_id"]])
            with patch.dict(os.environ, {"DOCQA_CONNECTOR_SOURCE_ACL": "{}"}):
                with self.assertRaises(ConnectorRevoked):
                    self.server._assert_current_connector_access(
                        {"workspace_id": "local", "principal_id": "alice"},
                        [source["source_id"]],
                    )
                redacted = client.get(f"/api/conversations/{answer['conversation_id']}",
                                      headers=alice).json()
                self.assertEqual(redacted["messages"][-1]["response"]["result_status"],
                                 "source_revoked")
                self.assertNotIn("30 days", str(redacted))

    def test_connector_chat_composes_upload_and_redacts_after_revocation(self):
        from langgraph_flow.graph_builder import build_graph
        self.stub_graph_models()
        self.server._graph = build_graph()
        self.stack.enter_context(patch.object(self.server, "generate_questions", return_value=[]))
        self.stack.enter_context(patch("retrieval.reranker.get_reranker_model", return_value=None))
        source_root = self.temp / "connector-root"
        (source_root / "procedures").mkdir(parents=True)
        (source_root / "procedures" / "gamma.txt").write_text("Gamma policy deadline is 45 days.")
        with patch.dict(os.environ, {"DOCQA_CONNECTOR_ROOT": str(source_root),
                                     "DOCQA_CONNECTOR_SOURCE_ACL": "{}"}):
            client = self.client()
            upload = client.post("/api/upload", files={"files": ("alpha.txt", b"Alpha policy deadline is 30 days.")})
            self.assertEqual(upload.status_code, 200, upload.text)
            base_version = upload.json()["corpus_version"]
            created = client.post("/api/connectors/sources", json={
                "source_key": "procedures", "display_name": "Procedures",
                "idempotency_key": "create-chat-0001",
            })
            self.assertEqual(created.status_code, 201, created.text)
            source = created.json()["source"]
            synced = client.post(f"/api/connectors/sources/{source['source_id']}/sync",
                                 headers={"idempotency-key": "sync-chat-0001"})
            self.assertEqual(synced.status_code, 200, synced.text)
            published = client.post(f"/api/connectors/sources/{source['source_id']}/publish")
            self.assertEqual(published.status_code, 200, published.text)
            self.assertEqual(published.json()["source"]["published_snapshot_id"],
                             synced.json()["sync"]["snapshot_id"])
            index_folders = self.temp / "vector_db" / "connector_indexes" / "corpora"
            self.assertEqual(len(list(index_folders.iterdir())), 1)

            state = client.get("/api/state").json()
            self.assertEqual(len(state["sources"]), 2)
            self.assertNotEqual(state["corpus_version"], base_version)
            gamma = client.post("/api/chat", json={"query": "What is the Gamma policy deadline?"})
            self.assertEqual(gamma.status_code, 200, gamma.text)
            answer = gamma.json()
            self.assertEqual(answer["connector_source_ids"], [source["source_id"]])
            self.assertTrue(any(item["origin"] == "connector" and
                                item["connector_snapshot_id"] == synced.json()["sync"]["snapshot_id"]
                                for item in answer["citations"]))
            self.assertTrue(any(doc["metadata"].get("connector_file_sha256")
                                for doc in answer["documents"]))
            alpha = client.post("/api/chat", json={"query": "What is the Alpha policy deadline?"})
            self.assertEqual(alpha.status_code, 200, alpha.text)
            self.assertTrue(any(doc["metadata"].get("source") == "alpha.txt"
                                for doc in alpha.json()["documents"]))
            updated_upload = client.post("/api/upload", files={
                "files": ("delta.txt", b"Delta policy deadline is 90 days.")
            })
            self.assertEqual(updated_upload.status_code, 200, updated_upload.text)
            self.assertEqual(len(client.get("/api/state").json()["sources"]), 2)
            retained = client.post("/api/chat", json={"query": "What is the Gamma policy deadline?"})
            self.assertEqual(retained.status_code, 200, retained.text)
            self.assertEqual(retained.json()["connector_source_ids"], [source["source_id"]])

            (source_root / "procedures" / "gamma.txt").write_text("Gamma policy deadline is 60 days.")
            changed = client.post(f"/api/connectors/sources/{source['source_id']}/sync",
                                  headers={"idempotency-key": "sync-chat-0002"})
            self.assertEqual(changed.status_code, 200, changed.text)
            current_upload_version = updated_upload.json()["corpus_version"]
            self.assertEqual(client.get("/api/state").json()["corpus_version"], current_upload_version)
            pending = client.get("/api/connectors/sources").json()["sources"][0]
            self.assertEqual(pending["publication_state"], "needs_publish")
            republished = client.post(f"/api/connectors/sources/{source['source_id']}/publish")
            self.assertEqual(republished.status_code, 200, republished.text)
            self.assertNotEqual(client.get("/api/state").json()["corpus_version"], current_upload_version)

            revoked = client.delete(f"/api/connectors/grants/{source['grant_id']}")
            self.assertEqual(revoked.status_code, 200, revoked.text)
            conversation = client.get(f"/api/conversations/{answer['conversation_id']}").json()
            self.assertEqual(conversation["messages"][-1]["response"]["result_status"], "source_revoked")
            self.assertNotIn("45 days", str(conversation))
            replay = client.get(f"/api/conversations/{answer['conversation_id']}/events").json()
            self.assertNotIn("45 days", str(replay))
            self.assertEqual(client.get("/api/state").json()["corpus_version"], current_upload_version)
            self.assertEqual(list(index_folders.iterdir()), [])

    def test_empty_snapshot_publish_removes_connector_from_chat(self):
        source_root = self.temp / "connector-root"
        folder = source_root / "policies"
        folder.mkdir(parents=True)
        (folder / "note.txt").write_text("Beta retention is five years.")
        with patch.dict(os.environ, {"DOCQA_CONNECTOR_ROOT": str(source_root),
                                     "DOCQA_CONNECTOR_SOURCE_ACL": "{}"}):
            client = self.client()
            source = client.post("/api/connectors/sources", json={
                "source_key": "policies", "display_name": "Policies",
                "idempotency_key": "create-empty-0001",
            }).json()["source"]
            path = f"/api/connectors/sources/{source['source_id']}"
            self.assertEqual(client.post(path+"/sync", headers={"idempotency-key": "sync-empty-0001"}).status_code, 200)
            self.assertEqual(client.post(path+"/publish").status_code, 200)
            self.assertTrue(client.get("/api/state").json()["vectorstore_ready"])
            (folder / "note.txt").unlink()
            self.assertEqual(client.post(path+"/sync", headers={"idempotency-key": "sync-empty-0002"}).status_code, 200)
            self.assertEqual(client.post(path+"/publish").status_code, 200)
            self.assertFalse(client.get("/api/state").json()["vectorstore_ready"])
            self.assertEqual(client.get("/api/connectors/sources").json()["sources"][0]["published_file_count"], 0)

    def test_failed_publication_leaves_existing_corpus_unchanged(self):
        source_root = self.temp / "connector-root"
        folder = source_root / "policies"
        folder.mkdir(parents=True)
        (folder / "note.txt").write_text("Beta retention is five years.")
        with patch.dict(os.environ, {"DOCQA_CONNECTOR_ROOT": str(source_root),
                                     "DOCQA_CONNECTOR_SOURCE_ACL": "{}"}):
            client = self.client()
            source = client.post("/api/connectors/sources", json={
                "source_key": "policies", "display_name": "Policies",
                "idempotency_key": "create-fail-0001",
            }).json()["source"]
            path = f"/api/connectors/sources/{source['source_id']}"
            self.assertEqual(client.post(path+"/sync", headers={"idempotency-key": "sync-fail-0001"}).status_code, 200)
            with patch("connectors.indexing.parse_in_isolated_process", side_effect=RuntimeError("private detail")):
                failed = client.post(path+"/publish")
            self.assertEqual(failed.status_code, 503)
            self.assertNotIn("private detail", failed.text)
            self.assertIsNone(client.get("/api/state").json()["corpus_version"])
            self.assertEqual(client.get("/api/connectors/sources").json()["sources"][0]["publication_state"],
                             "needs_publish")

    def test_register_sync_inspect_and_revoke_end_to_end(self):
        source_root = self.temp / "connector-root"
        (source_root / "sow").mkdir(parents=True)
        (source_root / "sow" / "scope.md").write_text("# Approved scope")
        environment = {
            "DOCQA_CONNECTOR_ROOT": str(source_root),
            "DOCQA_CONNECTOR_SOURCE_ACL": "{}",
            "DOCQA_DATA_DIR": str(self.temp / "connector-data"),
        }
        with patch.dict(os.environ, environment):
            client = self.client()
            created = client.post("/api/connectors/sources", json={
                "source_key": "sow", "display_name": "Current SOW",
                "idempotency_key": "create-api-0001",
            })
            self.assertEqual(created.status_code, 201, created.text)
            source = created.json()["source"]

            synced = client.post(
                f"/api/connectors/sources/{source['source_id']}/sync",
                headers={"idempotency-key": "sync-api-0001"},
            )
            self.assertEqual(synced.status_code, 200, synced.text)
            self.assertTrue(synced.json()["sync"]["changed"])
            listed = client.get("/api/connectors/sources").json()["sources"]
            self.assertEqual(listed[0]["freshness"], "fresh")
            snapshots = client.get(
                f"/api/connectors/sources/{source['source_id']}/snapshots",
            ).json()["snapshots"]
            self.assertEqual(snapshots[0]["files"][0]["relative_path"], "scope.md")

            revoked = client.delete(f"/api/connectors/grants/{source['grant_id']}")
            self.assertEqual(revoked.status_code, 200)
            blocked = client.get(
                f"/api/connectors/sources/{source['source_id']}/snapshots",
            )
            self.assertEqual(blocked.status_code, 409)
            self.assertEqual(blocked.json()["code"], "connector_revoked")
            removed = client.delete(f"/api/connectors/sources/{source['source_id']}")
            self.assertEqual(removed.status_code, 200)
            missing = client.get(
                f"/api/connectors/sources/{source['source_id']}/snapshots",
            )
            self.assertEqual(missing.status_code, 404)

    def test_connector_configuration_and_request_values_fail_safely(self):
        with patch.dict(os.environ, {"DOCQA_CONNECTOR_SOURCE_ACL": "{}"}):
            os.environ.pop("DOCQA_CONNECTOR_ROOT", None)
            response = self.client().post("/api/connectors/sources", json={
                "source_key": "sow", "display_name": "SOW",
                "idempotency_key": "create-api-0002",
            })
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "connector_not_configured")
        self.assertNotIn(str(self.temp), response.text)

        invalid = self.client().post("/api/connectors/sources", json={
            "source_key": "../private", "display_name": "Private",
            "idempotency_key": "create-api-0003",
        })
        self.assertEqual(invalid.status_code, 422)


if __name__ == "__main__":
    unittest.main()
