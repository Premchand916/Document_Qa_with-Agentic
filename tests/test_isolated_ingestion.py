"""Step 3 parser process and deadline checks."""

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
