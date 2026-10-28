"""Step 10 durable ingestion job and operational-boundary tests."""
import asyncio
import threading
import time
from unittest.mock import patch

from tests.support import OfflineCase, Upload


class IngestionJobRepositoryTests(OfflineCase):
    def repository(self, **options):
        from storage.ingestion_jobs import IngestionJobRepository
        return IngestionJobRepository(self.temp / "jobs", **options)

    def test_idempotent_submit_and_payload_conflict(self):
        from storage.ingestion_jobs import IngestionJobConflict
        repository = self.repository()
        first = repository.submit("alpha", "alice", "request-123", [("a.txt", b"alpha")], None)
        replay = repository.submit("alpha", "alice", "request-123", [("a.txt", b"alpha")], None)
        self.assertEqual(replay["job_id"], first["job_id"])
        with self.assertRaises(IngestionJobConflict):
            repository.submit("alpha", "alice", "request-123", [("a.txt", b"different")], None)

    def test_capacity_and_owner_boundaries(self):
        from storage.ingestion_jobs import IngestionCapacityError, IngestionJobNotFound
        repository = self.repository(max_pending_per_workspace=1)
        job = repository.submit("alpha", "alice", "request-123", [("a.txt", b"alpha")], None)
        with self.assertRaises(IngestionCapacityError):
            repository.submit("alpha", "bob", "request-456", [("b.txt", b"beta")], None)
        for workspace, principal in (("alpha", "bob"), ("beta", "alice")):
            with self.assertRaises(IngestionJobNotFound):
                repository.get(workspace, principal, job["job_id"])
            with self.assertRaises(IngestionJobNotFound):
                repository.request_cancel(workspace, principal, job["job_id"])

    def test_only_expired_lease_is_recovered_and_stale_worker_fails(self):
        from storage.ingestion_jobs import IngestionJobConflict
        repository = self.repository(lease_seconds=10)
        job = repository.submit("alpha", "alice", "request-123", [("a.txt", b"alpha")], None, now=100)
        old_token = repository.claim(job["job_id"], now=100)
        self.assertIsNotNone(old_token)
        self.assertEqual(repository.recover_expired(now=105), 0)
        self.assertIsNone(repository.claim(job["job_id"], now=105))
        with self.assertRaises(IngestionJobConflict):
            repository.complete(job["job_id"], old_token, {"bad": True}, now=111)
        self.assertEqual(repository.recover_expired(now=111), 1)
        new_token = repository.claim(job["job_id"], now=111)
        self.assertNotEqual(new_token, old_token)
        with self.assertRaises(IngestionJobConflict):
            repository.complete(job["job_id"], old_token, {"bad": True}, now=112)

    def test_cancellation_and_terminal_purge_remove_input_bytes(self):
        repository = self.repository(retention_days=1)
        job = repository.submit("alpha", "alice", "request-123", [("secret.txt", b"secret")], None, now=100)
        cancelled = repository.request_cancel("alpha", "alice", job["job_id"], now=101)
        self.assertEqual(cancelled["status"], "cancelled")
        with repository._connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM ingestion_inputs").fetchone()[0]
        self.assertEqual(count, 0)
        self.assertEqual(repository.purge_expired(now=100 + 86_401), 1)


class IngestionJobApiTests(OfflineCase):
    def setUp(self):
        super().setUp()
        self.stack.enter_context(patch.object(self.server, "generate_questions", return_value=[]))

    def persistent_client(self):
        from fastapi.testclient import TestClient
        return self.stack.enter_context(TestClient(
            self.server.app, raise_server_exceptions=False,
            client=("127.0.0.1", 12345), base_url="http://127.0.0.1",
        ))

    def wait_for_terminal(self, client, job_id, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = client.get(f"/api/ingestion-jobs/{job_id}")
            self.assertEqual(response.status_code, 200, response.text)
            job = response.json()
            if job["status"] in ("ready", "failed", "cancelled"):
                return job
            time.sleep(0.01)
        self.fail("Ingestion job did not reach a terminal state")

    def test_async_upload_exposes_status_timings_and_replays_one_job(self):
        client = self.persistent_client()
        headers = {"prefer": "respond-async", "idempotency-key": "upload-request-123"}
        first = client.post("/api/upload", headers=headers, files={"files": ("policy.txt", b"alpha policy")})
        self.assertEqual(first.status_code, 202, first.text)
        job_id = first.json()["job"]["job_id"]
        job = self.wait_for_terminal(client, job_id)
        self.assertEqual(job["status"], "ready")
        self.assertEqual(job["result"]["ingestion_job_id"], job_id)
        self.assertEqual(job["result"]["corpus_version"], job_id)
        self.assertIn("parsing", job["stage_durations_ms"])
        self.assertIsNotNone(job["queue_duration_ms"])
        replay = client.post("/api/upload", headers=headers, files={"files": ("policy.txt", b"alpha policy")})
        self.assertEqual(replay.status_code, 202)
        self.assertEqual(replay.json()["job"]["job_id"], job_id)
        conflict = client.post("/api/upload", headers=headers, files={"files": ("policy.txt", b"changed")})
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(conflict.json()["code"], "ingestion_job_conflict")

    def test_slow_job_allows_state_read_and_cancel_preserves_previous_corpus(self):
        client = self.persistent_client()
        old = client.post("/api/upload", files={"files": ("old.txt", b"alpha original")}).json()["corpus_version"]
        started = threading.Event()
        release = threading.Event()

        def slow_parse(name, content):
            started.set()
            release.wait(timeout=2)
            return self.server.parse_uploaded_file(Upload(name, content))

        headers = {"prefer": "respond-async", "idempotency-key": "upload-cancel-123"}
        with patch.object(self.server, "parse_in_isolated_process", side_effect=slow_parse):
            submitted = client.post("/api/upload", headers=headers,
                                    files={"files": ("new.txt", b"replacement")})
            job_id = submitted.json()["job"]["job_id"]
            self.assertTrue(started.wait(timeout=1))
            latencies = []
            for _ in range(20):
                before = time.monotonic()
                state = client.get("/api/state")
                latencies.append(time.monotonic() - before)
                self.assertEqual(state.status_code, 200)
            p95 = sorted(latencies)[18]
            self.assertLess(p95, 0.25)
            self.assertEqual(state.json()["corpus_version"], old)
            cancelled = client.delete(f"/api/ingestion-jobs/{job_id}")
            self.assertIn(cancelled.json()["status"], ("cancel_requested", "cancelled"))
            release.set()
            job = self.wait_for_terminal(client, job_id)
        self.assertEqual(job["status"], "cancelled")
        self.assertEqual(client.get("/api/state").json()["corpus_version"], old)

    def test_failure_is_safe_and_never_replaces_previous_corpus(self):
        client = self.persistent_client()
        old = client.post("/api/upload", files={"files": ("old.txt", b"alpha original")}).json()["corpus_version"]
        with patch.object(self.server, "build_vector_store",
                          side_effect=RuntimeError("SECRET internal embedding failure")):
            response = client.post("/api/upload", headers={"idempotency-key": "upload-failure-123"},
                                   files={"files": ("new.txt", b"replacement")})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["code"], "ingestion_failed")
        self.assertNotIn("SECRET", response.text)
        self.assertEqual(client.get("/api/state").json()["corpus_version"], old)


class IdempotentPublicationTests(OfflineCase):
    def test_retry_after_commit_reuses_same_corpus_version(self):
        from langchain_core.documents import Document
        repository = self.server.get_repository()
        vectorstore = self.store.build_vector_store([
            Document(page_content="alpha", metadata={"source": "a.txt"})
        ])
        version = "a" * 32
        original = repository._commit_publication

        def committed_then_lost(*args):
            original(*args)
            raise OSError("synthetic response loss")

        with patch.object(repository, "_commit_publication", side_effect=committed_then_lost):
            with self.assertRaises(OSError):
                repository.publish("local", vectorstore, [], [], ["txt"], [],
                                   expected_version=None, version=version)
        replay = self.server.get_repository().publish(
            "local", vectorstore, [], [], ["txt"], [], expected_version=None, version=version
        )
        self.assertEqual(replay, version)
        self.assertEqual(self.server.get_repository().state("local")["active_version"], version)

    def test_startup_recovers_and_completes_expired_job(self):
        from storage.ingestion_jobs import IngestionJobRepository
        repository = IngestionJobRepository(self.temp / "vector_db", lease_seconds=10)
        old_time = time.time() - 20
        job = repository.submit("local", "local-dev", "restart-request-123",
                                [("policy.txt", b"alpha recovered")], None, now=old_time)
        repository.claim(job["job_id"], now=old_time)

        async def restart():
            with patch.object(self.server, "get_ingestion_repository", return_value=repository), \
                    patch.object(self.server, "generate_questions", return_value=[]):
                await self.server._startup()
                await self.server._INGESTION_TASKS[job["job_id"]]

        asyncio.run(restart())
        restored = repository.get("local", "local-dev", job["job_id"])
        self.assertEqual(restored["status"], "ready")
        self.assertEqual(restored["result"]["corpus_version"], job["job_id"])
