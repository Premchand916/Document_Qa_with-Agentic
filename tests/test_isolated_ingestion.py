"""Step 3 parser process and deadline checks."""

import asyncio
import threading
import time
from unittest.mock import patch

from tests.support import OfflineCase


class IsolatedIngestionTests(OfflineCase):
    def test_csv_parse_returns_typed_table_and_documents(self):
        from ingestion.isolated_parser import parse_in_isolated_process
        result = parse_in_isolated_process("missing.csv", b"amount,region\n10,North\n,South\n30,North\n", timeout=15)
        self.assertEqual(result.tabular_assets[0]["dataframe"]["amount"].sum(), 40)
        self.assertEqual(str(result.tabular_assets[0]["dataframe"]["amount"].dtype), "Int64")
        self.assertTrue(any("North" in doc.page_content for doc in result.documents))

    def test_pdf_report_and_coordinates_cross_process_boundary(self):
        from ingestion.isolated_parser import parse_in_isolated_process
        from tests.support import text_pdf
        result = parse_in_isolated_process("policy.pdf", text_pdf("Alpha deadline 30 days"), timeout=15)
        self.assertEqual(result.extraction_report["pages_with_coordinates"], [1])
        self.assertEqual(result.documents[0].metadata["line_start"], 1)
        self.assertEqual(len(result.documents[0].metadata["bbox"]), 4)

    def test_timeout_terminates_parser_process(self):
        import multiprocessing
        from ingestion.isolated_parser import ParserTimeoutError, parse_in_isolated_process
        with self.assertRaises(ParserTimeoutError):
            parse_in_isolated_process("slow.txt", b"alpha", timeout=0.001)
        self.assertEqual(multiprocessing.active_children(), [])

    def test_parent_rejects_input_over_parser_byte_limit_before_spawn(self):
        import multiprocessing
        from ingestion.file_loader import IngestionLimitError
        import ingestion.isolated_parser as isolated
        with patch.object(isolated, "MAX_PARSE_INPUT_BYTES", 3):
            with self.assertRaises(IngestionLimitError):
                isolated.parse_in_isolated_process("large.txt", b"four", timeout=1)
        self.assertEqual(multiprocessing.active_children(), [])

    def test_parent_terminates_worker_over_resident_memory_limit(self):
        import multiprocessing
        import ingestion.isolated_parser as isolated
        with patch.object(isolated, "_worker_resident_bytes", return_value=isolated.MAX_WORKER_RSS + 1):
            with self.assertRaisesRegex(isolated.IngestionLimitError, "resident-memory"):
                isolated.parse_in_isolated_process("large.txt", b"alpha", timeout=5)
        self.assertEqual(multiprocessing.active_children(), [])

    def test_linux_worker_address_space_contract(self):
        import resource
        import ingestion.isolated_parser as isolated
        with patch.object(isolated.sys, "platform", "linux"), patch.object(resource, "setrlimit") as setter:
            isolated._limit_worker_memory()
        setter.assert_called_once_with(
            resource.RLIMIT_AS,
            (isolated.MAX_WORKER_ADDRESS_SPACE, isolated.MAX_WORKER_ADDRESS_SPACE),
        )

    def test_worker_rss_growth_is_relative_but_emergency_bounded(self):
        import ingestion.isolated_parser as isolated
        baseline = 100 * 1024 * 1024
        self.assertEqual(isolated._worker_rss_limit(baseline),
                         baseline + isolated.MAX_WORKER_RSS_GROWTH)
        self.assertEqual(isolated._worker_rss_limit(isolated.MAX_WORKER_RSS),
                         isolated.MAX_WORKER_RSS)

    def test_api_runs_at_most_two_ingestion_jobs_concurrently(self):
        active = 0
        peak = 0
        lock = threading.Lock()

        def process(*args):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return {"ready": True}

        class Repository:
            lease_seconds = 120

            def claim(self, job_id):
                return f"token-{job_id}"

            def complete(self, *args):
                pass

        async def exercise():
            repository = Repository()
            with patch.object(self.server, "_INGESTION_SLOTS", asyncio.Semaphore(2)), \
                    patch.object(self.server, "get_ingestion_repository", return_value=repository), \
                    patch.object(self.server, "_process_ingestion_job", side_effect=process):
                await asyncio.gather(*(self.server._run_ingestion_job(str(index)) for index in range(4)))

        asyncio.run(exercise())
        self.assertEqual(peak, 2)

    def test_api_timeout_preserves_previous_corpus(self):
        from ingestion.isolated_parser import ParserTimeoutError
        client = self.client()
        original = client.post("/api/upload", files={"files": ("old.txt", b"alpha original")})
        self.assertEqual(original.status_code, 200)
        with patch.object(self.server, "parse_in_isolated_process", side_effect=ParserTimeoutError()):
            timeout = client.post("/api/upload", files={"files": ("new.txt", b"replacement")})
        self.assertEqual(timeout.status_code, 504)
        self.assertEqual(timeout.json()["code"], "parser_timeout")
        self.assertEqual(client.get("/api/state").json()["corpus_version"], original.json()["corpus_version"])
