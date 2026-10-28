import json
from io import BytesIO
import os
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

from tests.support import FIXTURES, OfflineCase, Upload, text_pdf


class IngestionTests(OfflineCase):
    @staticmethod
    def scanned_pdf(page_count=1):
        from PIL import Image, ImageDraw
        pages = []
        for index in range(page_count):
            image = Image.new("RGB", (900, 260), "white")
            ImageDraw.Draw(image).text((40, 100), f"SCANNED POLICY PAGE {index + 1}", fill="black")
            pages.append(image)
        output = BytesIO()
        pages[0].save(output, format="PDF", save_all=True, append_images=pages[1:], resolution=150)
        return output.getvalue()

    def test_utf8_text_retains_content_and_source(self):
        from ingestion.file_loader import load_uploaded_file
        raw = (FIXTURES / "corpus" / "policy_es.txt").read_bytes()
        docs = load_uploaded_file(Upload("policy_es.txt", raw))
        self.assertEqual(docs[0].page_content, raw.decode("utf-8"))
        self.assertEqual(docs[0].metadata["source"], "policy_es.txt")

    def test_real_pdf_parser_extracts_text_and_page(self):
        from ingestion.file_loader import parse_uploaded_file
        text = (FIXTURES / "corpus" / "policy.txt").read_text().strip()
        parsed = parse_uploaded_file(Upload("policy.pdf", text_pdf(text)))
        docs = parsed.documents
        self.assertIn("30 days", docs[0].page_content)
        self.assertEqual(docs[0].metadata["source"], "policy.pdf")
        self.assertEqual(str(docs[0].metadata["page"]), "1")
        self.assertEqual(docs[0].metadata["content_type"], "pdf_block")
        x0, top, x1, bottom = docs[0].metadata["bbox"]
        self.assertTrue(0 <= x0 < x1 <= 612 and 0 <= top < bottom <= 792)
        self.assertEqual(parsed.extraction_report["pages_with_coordinates"], [1])
        self.assertEqual(parsed.extraction_report["pages_without_text"], [])

    def test_csv_preserves_complete_numeric_values_and_provenance(self):
        from ingestion.file_loader import extract_tabular_assets, load_uploaded_file
        raw = (FIXTURES / "corpus" / "sales.csv").read_bytes()
        docs = load_uploaded_file(Upload("sales.csv", raw))
        assets = extract_tabular_assets(Upload("sales.csv", raw))
        self.assertEqual(assets[0]["dataframe"]["amount"].sum(), 30)
        self.assertTrue(all(d.metadata["source"] == "sales.csv" for d in docs))
        self.assertTrue(any("North" in d.page_content for d in docs))

    def test_json_leaves_keep_escaped_paths_and_values(self):
        from ingestion.file_loader import load_uploaded_file
        docs = load_uploaded_file(Upload("policy.json", b'{"policy/a": {"deadline": 30}, "revisions": [30, null]}'))
        by_path = {doc.metadata["json_pointer"]: doc for doc in docs}
        self.assertEqual(set(by_path), {"/policy~1a/deadline", "/revisions/0", "/revisions/1"})
        self.assertIn("30", by_path["/policy~1a/deadline"].page_content)
        self.assertIn("null", by_path["/revisions/1"].page_content)
        self.assertTrue(all(doc.metadata["source"] == "policy.json" for doc in docs))
        from ingestion.semantic_chunker import semantic_chunk_documents
        chunked = semantic_chunk_documents(docs)
        ids = {doc.metadata["json_pointer"]: doc.metadata["chunk_id"] for doc in chunked}
        self.assertNotEqual(ids["/policy~1a/deadline"], ids["/revisions/0"])

    def test_json_node_and_depth_limits(self):
        from ingestion.file_loader import IngestionLimitError, parse_uploaded_file
        import ingestion.file_loader as loader
        with patch.object(loader, "MAX_JSON_NODES", 2):
            with self.assertRaises(IngestionLimitError):
                parse_uploaded_file(Upload("wide.json", b'{"a": 1, "b": 2}'))
        with patch.object(loader, "MAX_JSON_DEPTH", 1):
            with self.assertRaises(IngestionLimitError):
                parse_uploaded_file(Upload("deep.json", b'{"a": {"b": 2}}'))

    def test_long_text_splits_and_retains_last_fact_and_metadata(self):
        from ingestion.semantic_chunker import semantic_chunk_documents
        from langchain_core.documents import Document
        docs = semantic_chunk_documents([Document(page_content="alpha details " * 200 + "FINAL_FACT_42", metadata={"source": "long.txt", "page": 1})])
        self.assertGreater(len(docs), 1)
        self.assertTrue(any("FINAL_FACT_42" in d.page_content for d in docs))
        self.assertTrue(all(len(d.page_content) <= 800 for d in docs))
        self.assertTrue(all(d.metadata["source"] == "long.txt" and d.metadata["page"] == 1 and d.metadata["chunk_id"] for d in docs))

    def test_empty_supported_file_is_rejected(self):
        from ingestion.file_loader import load_uploaded_file
        with self.assertRaisesRegex(ValueError, "No readable content"):
            load_uploaded_file(Upload("empty.txt", b"  "))

    def test_nullable_numeric_csv_stays_numeric_for_calculations(self):
        from ingestion.file_loader import parse_uploaded_file
        result = parse_uploaded_file(Upload("missing.csv", b"amount,region\n10,North\n,South\n30,North\n"))
        frame = result.tabular_assets[0]["dataframe"]
        self.assertEqual(frame["amount"].sum(), 40)
        self.assertEqual(str(frame["amount"].dtype), "Int64")
        self.assertTrue(frame["amount"].isna().iloc[1])
        self.assertTrue(any("Row 3" in doc.page_content for doc in result.documents))

    def test_tabular_file_is_parsed_once(self):
        import ingestion.file_loader as loader
        with patch.object(loader, "_read_delimited_frame", wraps=loader._read_delimited_frame) as reader:
            result = loader.parse_uploaded_file(Upload("sales.csv", b"amount\n10\n20\n"))
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(result.tabular_assets[0]["dataframe"]["amount"].sum(), 30)

    def test_long_json_and_slide_chunks_retain_tail_and_stable_ids(self):
        from langchain_core.documents import Document
        from ingestion.semantic_chunker import semantic_chunk_documents
        raw = [Document(page_content="alpha " * 1000 + "FINAL_FACT_42", metadata={"source": "large.json", "source_id": "source-1", "page": "JSON", "content_type": "json_document"}),
               Document(page_content="slide " * 1000 + "SLIDE_TAIL", metadata={"source": "deck.pptx", "source_id": "source-2", "page": "Slide 1", "content_type": "slide"})]
        first = semantic_chunk_documents(raw)
        second = semantic_chunk_documents(raw)
        self.assertTrue(any("FINAL_FACT_42" in doc.page_content for doc in first))
        self.assertTrue(any("SLIDE_TAIL" in doc.page_content for doc in first))
        self.assertEqual([doc.metadata["chunk_id"] for doc in first], [doc.metadata["chunk_id"] for doc in second])
        self.assertEqual(len({doc.metadata["chunk_id"] for doc in first}), len(first))
        self.assertTrue(all(len(doc.page_content.encode("utf-8")) <= 800 for doc in first))

    def test_scanned_pdf_is_reported_as_extraction_failure(self):
        from ingestion.file_loader import parse_uploaded_file
        from types import SimpleNamespace
        pdf = MagicMock()
        pdf.__enter__.return_value.pages = [SimpleNamespace(extract_text=lambda: None)]
        with patch("ingestion.file_loader.pdfplumber.open", return_value=pdf):
            with self.assertRaisesRegex(ValueError, "OCR required"):
                parse_uploaded_file(Upload("scan.pdf", b"%PDF-1.4"))

    def test_scanned_pdf_uses_bounded_local_ocr_with_provenance(self):
        import ingestion.file_loader as loader
        with patch.object(loader, "_tesseract_text", return_value="ALPHA DEADLINE 30 DAYS") as ocr:
            parsed = loader.parse_uploaded_file(Upload("scan.pdf", self.scanned_pdf()))
        self.assertEqual(ocr.call_count, 1)
        self.assertIn("30 DAYS", parsed.documents[0].page_content)
        self.assertEqual(parsed.documents[0].metadata["content_type"], "ocr_text")
        self.assertEqual(parsed.documents[0].metadata["ocr_engine"], "tesseract")
        self.assertEqual(parsed.extraction_report["ocr_pages"], [1])
        self.assertEqual(parsed.extraction_report["pages_without_text"], [])

    def test_pdf_ocr_page_budget_leaves_explicit_partial_warning(self):
        import ingestion.file_loader as loader
        with patch.object(loader, "MAX_PDF_OCR_PAGES", 2), \
                patch.object(loader, "_tesseract_text", side_effect=["PAGE ONE", "PAGE TWO"]):
            parsed = loader.parse_uploaded_file(Upload("many-scans.pdf", self.scanned_pdf(4)))
        self.assertEqual(parsed.extraction_report["ocr_pages"], [1, 2])
        self.assertEqual(parsed.extraction_report["pages_without_text"], [3, 4])
        self.assertIn("3, 4", parsed.warnings[0])

    def test_image_ocr_enforces_pixel_frame_and_timeout_limits(self):
        import ingestion.file_loader as loader
        from PIL import Image
        image = Image.new("RGB", (100, 100), "white")
        encoded = BytesIO()
        image.save(encoded, format="PNG")
        with patch.object(loader, "MAX_IMAGE_PIXELS", 9_999):
            with self.assertRaises(loader.IngestionLimitError):
                loader.parse_uploaded_file(Upload("large.png", encoded.getvalue()))

        first, second = Image.new("RGB", (10, 10), "white"), Image.new("RGB", (10, 10), "white")
        multi = BytesIO()
        first.save(multi, format="TIFF", save_all=True, append_images=[second])
        with patch.object(loader, "MAX_IMAGE_FRAMES", 1):
            with self.assertRaises(loader.IngestionLimitError):
                loader.parse_uploaded_file(Upload("multi.tiff", multi.getvalue()))

        with patch("pytesseract.image_to_string", return_value="OCR TEXT") as engine:
            parsed = loader.parse_uploaded_file(Upload("text.png", encoded.getvalue()))
        self.assertEqual(parsed.extraction_report["ocr_engine"], "tesseract")
        self.assertEqual(parsed.extraction_report["ocr_pages"], [1])
        self.assertEqual(engine.call_args.kwargs["timeout"], loader.OCR_PAGE_TIMEOUT)

    def test_ocr_preprocessing_corrects_rotation_and_removes_isolated_noise(self):
        import ingestion.file_loader as loader
        from PIL import Image
        rotated = Image.new("L", (100, 50), 255)
        with patch("pytesseract.image_to_osd", return_value={"rotate": 90}):
            prepared, metadata = loader._prepare_ocr_image(rotated)
        self.assertEqual(prepared.size, (50, 100))
        self.assertEqual(metadata["ocr_rotation_degrees"], 90)
        self.assertTrue(metadata["ocr_preprocessed"])

        noisy = Image.new("L", (9, 9), 255)
        noisy.putpixel((4, 4), 0)
        with patch.dict(os.environ, {"DOCQA_OCR_AUTO_ROTATE": "false"}):
            cleaned, _ = loader._prepare_ocr_image(noisy)
        self.assertEqual(cleaned.getpixel((4, 4)), 255)

    def test_requested_ocr_languages_must_be_installed(self):
        import ingestion.file_loader as loader
        with patch.dict(os.environ, {"DOCQA_OCR_LANGUAGES": "eng+hin"}), \
                patch.object(loader, "_installed_tesseract_languages", return_value=frozenset({"eng"})):
            with self.assertRaisesRegex(loader.OCRConfigurationError, "hin"):
                loader._ocr_languages()

    def test_cloud_ocr_requires_explicit_opt_in(self):
        import ingestion.file_loader as loader
        from PIL import Image
        image = Image.new("RGB", (20, 20), "white")
        encoded = BytesIO()
        image.save(encoded, format="PNG")
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "configured-but-not-authorized",
                                     "DOCQA_ALLOW_CLOUD_OCR": ""}), \
                patch.object(loader, "_tesseract_text", return_value=""):
            with self.assertRaisesRegex(ValueError, "Could not extract text"):
                loader.parse_uploaded_file(Upload("blank.png", encoded.getvalue()))

    def test_partial_pdf_reports_blank_page_without_dropping_text(self):
        from types import SimpleNamespace
        from ingestion.file_loader import parse_uploaded_file
        pdf = MagicMock()
        pdf.__enter__.return_value.pages = [SimpleNamespace(extract_text=lambda: "Alpha policy"), SimpleNamespace(extract_text=lambda: None)]
        with patch("ingestion.file_loader.pdfplumber.open", return_value=pdf):
            result = parse_uploaded_file(Upload("partial.pdf", b"%PDF-1.4"))
        self.assertEqual(len(result.documents), 1)
        self.assertIn("page 2", result.warnings[0])
        self.assertEqual(result.extraction_report["pages_without_text"], [2])
        self.assertEqual(result.extraction_report["text_pages"], 1)

    def test_pdf_text_blocks_keep_line_ranges_and_stop_at_limit(self):
        from types import SimpleNamespace
        import ingestion.file_loader as loader
        lines = [{"text": f"policy line {index}", "x0": 50.0, "top": float(index * 12),
                  "x1": 180.0, "bottom": float(index * 12 + 10)} for index in range(9)]
        page = SimpleNamespace(extract_text_lines=lambda **kwargs: lines, width=612, height=792)
        pdf = MagicMock()
        pdf.__enter__.return_value.pages = [page]
        with patch.object(loader.pdfplumber, "open", return_value=pdf):
            parsed = loader.parse_uploaded_file(Upload("lines.pdf", b"%PDF-1.4"))
        self.assertEqual([(doc.metadata["line_start"], doc.metadata["line_end"]) for doc in parsed.documents],
                         [(1, 8), (9, 9)])
        self.assertEqual(parsed.documents[1].metadata["bbox"], [50.0, 96.0, 180.0, 106.0])
        with patch.object(loader.pdfplumber, "open", return_value=pdf), patch.object(loader, "MAX_PDF_TEXT_LINES", 8):
            with self.assertRaises(loader.IngestionLimitError):
                loader.parse_uploaded_file(Upload("lines.pdf", b"%PDF-1.4"))

    def test_table_and_archive_expansion_limits(self):
        from ingestion.file_loader import IngestionLimitError, parse_uploaded_file
        with patch("ingestion.file_loader.MAX_TABLE_ROWS", 2):
            with self.assertRaises(IngestionLimitError):
                parse_uploaded_file(Upload("large.csv", b"amount\n1\n2\n3\n"))
        buffer = BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("large.xml", "x" * 100)
        with patch("ingestion.file_loader.MAX_ARCHIVE_EXPANDED_BYTES", 50):
            with self.assertRaises(IngestionLimitError):
                parse_uploaded_file(Upload("large.xlsx", buffer.getvalue()))

    def test_csv_reader_stops_after_limit_plus_one_row(self):
        import ingestion.file_loader as loader
        raw = Upload("large.csv", b"amount\n1\n2\n3\n4\n")
        with patch.object(loader, "MAX_TABLE_ROWS", 2), patch.object(loader.pd, "read_csv", wraps=loader.pd.read_csv) as reader:
            with self.assertRaises(loader.IngestionLimitError):
                loader.parse_uploaded_file(raw)
        self.assertEqual(reader.call_args.kwargs["nrows"], 3)
