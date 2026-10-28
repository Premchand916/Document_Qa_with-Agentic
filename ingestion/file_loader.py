import json
import os
import hashlib
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import pandas as pd
import pdfplumber
from langchain_core.documents import Document
from pptx import Presentation

SUPPORTED_FILE_TYPES = {
    ".pdf": "PDF document",
    ".pptx": "PowerPoint presentation",
    ".xlsx": "Excel workbook",
    ".xlsm": "Excel macro workbook",
    ".csv": "CSV dataset",
    ".tsv": "TSV dataset",
    ".txt": "Text document",
    ".md": "Markdown document",
    ".json": "JSON document",
    # Image formats (OCR)
    ".jpg": "JPEG image",
    ".jpeg": "JPEG image",
    ".png": "PNG image",
    ".tiff": "TIFF image",
    ".tif": "TIFF image",
    ".bmp": "BMP image",
    ".webp": "WebP image",
}

TABULAR_EXTENSIONS = {".xlsx", ".xlsm", ".csv", ".tsv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
MAX_TABLE_ROWS = 100_000
MAX_PDF_PAGES = 500
MAX_PDF_TEXT_LINES = 20_000
PDF_LINES_PER_BLOCK = 8
MAX_SLIDES = 500
MAX_ARCHIVE_ENTRIES = 500
MAX_ARCHIVE_EXPANDED_BYTES = 100 * 1024 * 1024
MAX_JSON_NODES = 5000
MAX_JSON_DEPTH = 64
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_FRAMES = 10
MAX_OCR_CHARS = 1_000_000
MAX_PDF_OCR_PAGES = 5
PDF_OCR_RESOLUTION = 200
OCR_PAGE_TIMEOUT = 4


class IngestionLimitError(ValueError):
    """Supported input exceeds a parser resource limit."""


class OCRRequiredError(ValueError):
    """A PDF has no extractable text and needs OCR before indexing."""


@dataclass
class ParsedFile:
    documents: list[Document]
    tabular_assets: list[dict]
    warnings: list[str]
    extraction_report: dict = field(default_factory=dict)


def _to_bytes(uploaded_file):
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()

    if hasattr(uploaded_file, "read"):
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        return uploaded_file.read()

    raise TypeError("Unsupported uploaded file object.")


def _clean_cell(value):
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").strip()


def _normalize_dataframe(dataframe):
    normalized_df = dataframe.copy()
    normalized_df.columns = [str(column).strip() for column in normalized_df.columns]
    if len(normalized_df.index) > MAX_TABLE_ROWS:
        raise IngestionLimitError(f"Table exceeds {MAX_TABLE_ROWS} rows")
    return normalized_df.convert_dtypes()


def _check_archive(raw: bytes):
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        entries = archive.infolist()
        if len(entries) > MAX_ARCHIVE_ENTRIES or sum(item.file_size for item in entries) > MAX_ARCHIVE_EXPANDED_BYTES:
            raise IngestionLimitError("Office archive expansion limit exceeded")


def _read_excel_frames(file_obj):
    workbook_bytes = BytesIO(_to_bytes(file_obj))
    excel_file = pd.ExcelFile(workbook_bytes, engine="openpyxl")
    frames = []
    for sheet_name in excel_file.sheet_names:
        dataframe = excel_file.parse(sheet_name=sheet_name, nrows=MAX_TABLE_ROWS + 1)
        frames.append((sheet_name, _normalize_dataframe(dataframe)))
    return frames


def _read_delimited_frame(file_obj, extension):
    delimiter = "\t" if extension == ".tsv" else ","
    dataframe = pd.read_csv(BytesIO(_to_bytes(file_obj)), sep=delimiter, nrows=MAX_TABLE_ROWS + 1)
    return _normalize_dataframe(dataframe)


def _tabular_chunk_documents(dataframe, source_name, file_type, sheet_name="Data", chunk_size=20):
    documents = []
    normalized_df = _normalize_dataframe(dataframe)

    column_list = [column for column in normalized_df.columns if column]
    row_count = len(normalized_df.index)

    summary = (
        f"Source: {source_name}\n"
        f"Sheet: {sheet_name}\n"
        f"File type: {file_type}\n"
        f"Columns: {', '.join(column_list) if column_list else 'No columns detected'}\n"
        f"Row count: {row_count}"
    )
    documents.append(
        Document(
            page_content=summary,
            metadata={
                "source": source_name,
                "page": f"Sheet {sheet_name}",
                "sheet": sheet_name,
                "file_type": file_type,
                "content_type": "table_summary",
            },
        )
    )

    for start_index in range(0, row_count, chunk_size):
        end_index = min(start_index + chunk_size, row_count)
        chunk_df = normalized_df.iloc[start_index:end_index]
        row_lines = []
        for row_number, (_, row) in enumerate(chunk_df.iterrows(), start=start_index + 1):
            cell_pairs = []
            for column in column_list:
                cleaned = _clean_cell(row[column])
                if cleaned:
                    cell_pairs.append(f"{column}: {cleaned}")
            if cell_pairs:
                row_lines.append(f"Row {row_number}: " + " | ".join(cell_pairs))

        if not row_lines:
            continue

        content = (
            f"Source: {source_name}\n"
            f"Sheet: {sheet_name}\n"
            f"Columns: {', '.join(column_list) if column_list else 'No columns detected'}\n"
            f"Rows: {start_index + 1}-{end_index}\n"
            + "\n".join(row_lines)
        )
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": source_name,
                    "page": f"Sheet {sheet_name} Rows {start_index + 1}-{end_index}",
                    "sheet": sheet_name,
                    "row_start": start_index + 1,
                    "row_end": end_index,
                    "file_type": file_type,
                    "content_type": "table_rows",
                },
            )
        )

    return documents


def _tesseract_text(image) -> str:
    import pytesseract

    width, height = image.size
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise IngestionLimitError(f"Image exceeds {MAX_IMAGE_PIXELS} pixels")
    text = pytesseract.image_to_string(image, config="--psm 6", timeout=OCR_PAGE_TIMEOUT)
    if len(text) > MAX_OCR_CHARS:
        raise IngestionLimitError(f"OCR output exceeds {MAX_OCR_CHARS} characters")
    return text.strip()


def _ocr_pdf_pages(raw: bytes, source_name: str, page_numbers: list[int]) -> tuple[list[Document], list[int]]:
    documents = []
    failed = []
    try:
        with pdfplumber.open(BytesIO(raw)) as pdf:
            for page_number in page_numbers[:MAX_PDF_OCR_PAGES]:
                page = pdf.pages[page_number - 1]
                try:
                    scaled_pixels = int(float(page.width) * PDF_OCR_RESOLUTION / 72) * int(float(page.height) * PDF_OCR_RESOLUTION / 72)
                    if scaled_pixels > MAX_IMAGE_PIXELS:
                        raise IngestionLimitError(f"Rendered PDF page exceeds {MAX_IMAGE_PIXELS} pixels")
                    image = page.to_image(resolution=PDF_OCR_RESOLUTION).original
                    text = _tesseract_text(image)
                except IngestionLimitError:
                    raise
                except Exception:
                    failed.append(page_number)
                    continue
                if not text:
                    failed.append(page_number)
                    continue
                documents.append(Document(page_content=text, metadata={
                    "source": source_name, "page": page_number, "file_type": "pdf",
                    "content_type": "ocr_text", "ocr_engine": "tesseract",
                    "coordinate_status": "raster_page", "ocr_resolution_dpi": PDF_OCR_RESOLUTION,
                    "page_size": [float(page.width), float(page.height)],
                }))
    except IngestionLimitError:
        raise
    except Exception:
        return [], list(page_numbers[:MAX_PDF_OCR_PAGES])
    return documents, failed


def load_pdf(file_obj, source_name, warnings=None, report=None):
    documents = []
    blank_pages = []
    coordinate_pages = []
    text_pages = 0
    line_count = 0
    raw = _to_bytes(file_obj)
    with pdfplumber.open(BytesIO(raw)) as pdf:
        if len(pdf.pages) > MAX_PDF_PAGES:
            raise IngestionLimitError(f"PDF exceeds {MAX_PDF_PAGES} pages")
        if report is not None:
            report["page_count"] = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            extractor = getattr(page, "extract_text_lines", None)
            lines = extractor(return_chars=False) if callable(extractor) else []
            lines = [line for line in lines if line.get("text", "").strip()]
            line_count += len(lines)
            if line_count > MAX_PDF_TEXT_LINES:
                raise IngestionLimitError("PDF text-line limit exceeded")
            if lines:
                text_pages += 1
                coordinate_pages.append(page_num)
                for start in range(0, len(lines), PDF_LINES_PER_BLOCK):
                    block = lines[start:start + PDF_LINES_PER_BLOCK]
                    bbox = [min(float(line["x0"]) for line in block),
                            min(float(line["top"]) for line in block),
                            max(float(line["x1"]) for line in block),
                            max(float(line["bottom"]) for line in block)]
                    documents.append(Document(
                        page_content="\n".join(line["text"].strip() for line in block),
                        metadata={"source": source_name, "page": page_num, "file_type": "pdf",
                                  "content_type": "pdf_block", "line_start": start + 1,
                                  "line_end": start + len(block), "bbox": bbox,
                                  "page_size": [float(page.width), float(page.height)]},
                    ))
                continue
            text = page.extract_text()
            if text and text.strip():
                text_pages += 1
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": source_name,
                            "page": page_num,
                            "file_type": "pdf",
                            "content_type": "pdf_page",
                            "coordinate_status": "unavailable",
                        },
                    )
                )
            else:
                blank_pages.append(page_num)
    ocr_documents = []
    if blank_pages:
        ocr_documents, _ = _ocr_pdf_pages(raw, source_name, blank_pages)
        documents.extend(ocr_documents)
    ocr_pages = {int(document.metadata["page"]) for document in ocr_documents}
    remaining_blank_pages = [page for page in blank_pages if page not in ocr_pages]
    if report is not None:
        report.update(text_pages=text_pages + len(ocr_documents), pages_without_text=remaining_blank_pages,
                      pages_with_coordinates=coordinate_pages, ocr_pages=sorted(ocr_pages))
    if remaining_blank_pages and documents and warnings is not None:
        pages = ", ".join(str(page) for page in remaining_blank_pages[:20])
        suffix = "…" if len(remaining_blank_pages) > 20 else ""
        warnings.append(f"{source_name}: PDF page{'s' if len(remaining_blank_pages) != 1 else ''} {pages}{suffix} had no extractable text; OCR may be required.")
    return documents


def load_pptx(file_obj, source_name, warnings=None):
    documents = []
    blank_slides = 0
    presentation = Presentation(BytesIO(_to_bytes(file_obj)))
    if len(presentation.slides) > MAX_SLIDES:
        raise IngestionLimitError(f"Presentation exceeds {MAX_SLIDES} slides")

    for slide_index, slide in enumerate(presentation.slides, start=1):
        text_parts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                cleaned = shape.text.strip()
                if cleaned:
                    text_parts.append(cleaned)

        if not text_parts:
            blank_slides += 1
            continue

        slide_text = "\n".join(text_parts)
        documents.append(
            Document(
                page_content=slide_text,
                metadata={
                    "source": source_name,
                    "page": f"Slide {slide_index}",
                    "slide": slide_index,
                    "file_type": "pptx",
                    "content_type": "slide",
                },
            )
        )

    if blank_slides and documents and warnings is not None:
        unit = "slide" if blank_slides == 1 else "slides"
        warnings.append(f"{source_name}: {blank_slides} {unit} had no extractable text.")
    return documents


def load_excel(file_obj, source_name, extension):
    documents = []
    for sheet_name, dataframe in _read_excel_frames(file_obj):
        documents.extend(_tabular_chunk_documents(dataframe, source_name, extension.lstrip("."), sheet_name))

    return documents


def load_delimited_text(file_obj, source_name, extension):
    dataframe = _read_delimited_frame(file_obj, extension)
    return _tabular_chunk_documents(dataframe, source_name, extension.lstrip("."), "Data")


def load_text_document(file_obj, source_name, extension):
    text = _to_bytes(file_obj).decode("utf-8", errors="ignore")
    if not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": source_name,
                "page": extension.lstrip(".").upper(),
                "file_type": extension.lstrip("."),
                "content_type": "text_document",
            },
        )
    ]


def load_image_ocr(file_obj, source_name):
    """Extract text from an image via OCR.

    Tries pytesseract first (requires system Tesseract install). Gemini Vision
    requires both GOOGLE_API_KEY and explicit DOCQA_ALLOW_CLOUD_OCR=true.
    Raises ValueError when no authorized engine produces text.
    """
    image_bytes = _to_bytes(file_obj)
    extension = Path(source_name).suffix.lower()

    # ── pytesseract ───────────────────────────────────────────────────────────
    try:
        from PIL import Image
        import pytesseract

        image = Image.open(BytesIO(image_bytes))
        frame_count = int(getattr(image, "n_frames", 1))
        if frame_count > MAX_IMAGE_FRAMES:
            raise IngestionLimitError(f"Image exceeds {MAX_IMAGE_FRAMES} frames")
        documents = []
        total_chars = 0
        for frame in range(frame_count):
            image.seek(frame)
            if image.width <= 0 or image.height <= 0 or image.width * image.height > MAX_IMAGE_PIXELS:
                raise IngestionLimitError(f"Image exceeds {MAX_IMAGE_PIXELS} pixels")
            text = _tesseract_text(image.convert("RGB"))
            total_chars += len(text)
            if total_chars > MAX_OCR_CHARS:
                raise IngestionLimitError(f"OCR output exceeds {MAX_OCR_CHARS} characters")
            if text:
                documents.append(Document(page_content=text, metadata={
                    "source": source_name, "page": str(frame + 1),
                    "file_type": extension.lstrip("."), "content_type": "ocr_text",
                    "ocr_engine": "tesseract", "image_size": list(image.size),
                }))
        if documents:
            return documents
    except ImportError:
        pass
    except IngestionLimitError:
        raise
    except Exception:
        pass

    # ── Gemini Vision fallback ────────────────────────────────────────────────
    api_key = os.getenv("GOOGLE_API_KEY", "")
    cloud_ocr_enabled = os.getenv("DOCQA_ALLOW_CLOUD_OCR", "").strip().lower() in {"1", "true", "yes"}
    if api_key and cloud_ocr_enabled:
        try:
            import base64
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                        ".tiff": "image/tiff", ".tif": "image/tiff",
                        ".bmp": "image/bmp", ".webp": "image/webp"}
            mime_type = mime_map.get(extension, "image/png")
            img_b64 = base64.b64encode(image_bytes).decode()
            response = model.generate_content([
                "Extract all readable text from this image. Return only the extracted text.",
                {"inline_data": {"mime_type": mime_type, "data": img_b64}},
            ])
            text = response.text.strip()
            if text:
                return [
                    Document(
                        page_content=text,
                        metadata={
                            "source": source_name,
                            "page": "1",
                            "file_type": extension.lstrip("."),
                            "content_type": "ocr_text",
                            "ocr_engine": "gemini",
                        },
                    )
                ]
        except Exception:
            pass

    raise ValueError(
        f"Could not extract text from '{source_name}'. "
        "Install Tesseract + pytesseract, or explicitly enable cloud OCR with "
        "GOOGLE_API_KEY and DOCQA_ALLOW_CLOUD_OCR=true."
    )


def load_json_document(file_obj, source_name):
    raw_text = _to_bytes(file_obj).decode("utf-8", errors="ignore")
    parsed = json.loads(raw_text)
    documents = []
    stack = [("", parsed, 0)]
    visited = 0
    while stack:
        pointer, value, depth = stack.pop()
        visited += 1
        if visited > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            raise IngestionLimitError("JSON structure limit exceeded")
        if isinstance(value, dict) and value:
            if visited + len(stack) + len(value) > MAX_JSON_NODES:
                raise IngestionLimitError("JSON structure limit exceeded")
            for key, child in reversed(list(value.items())):
                escaped = key.replace("~", "~0").replace("/", "~1")
                stack.append((f"{pointer}/{escaped}", child, depth + 1))
            continue
        if isinstance(value, list) and value:
            if visited + len(stack) + len(value) > MAX_JSON_NODES:
                raise IngestionLimitError("JSON structure limit exceeded")
            for index in range(len(value) - 1, -1, -1):
                stack.append((f"{pointer}/{index}", value[index], depth + 1))
            continue
        rendered = json.dumps(value, ensure_ascii=False)
        documents.append(Document(
            page_content=f"Source: {source_name}\nJSON pointer: {pointer or '(root)'}\nValue: {rendered}",
            metadata={"source": source_name, "page": "JSON", "file_type": "json",
                      "content_type": "json_value", "json_pointer": pointer},
        ))
    return documents


def parse_uploaded_file(uploaded_file) -> ParsedFile:
    source_name = getattr(uploaded_file, "name", "uploaded_file")
    extension = Path(source_name).suffix.lower()

    if extension not in SUPPORTED_FILE_TYPES:
        raise ValueError(
            f"Unsupported file type '{extension}'. Supported types: {', '.join(sorted(SUPPORTED_FILE_TYPES))}"
        )

    raw = _to_bytes(uploaded_file)
    file_obj = BytesIO(raw)
    if extension in {".xlsx", ".xlsm", ".pptx"}:
        _check_archive(raw)
    assets = []
    warnings = []
    report = {"source": source_name, "file_type": extension.lstrip("."), "text_extracted": True,
              "ocr_engine": None, "page_count": None, "text_pages": None,
              "pages_without_text": [], "pages_with_coordinates": [], "ocr_pages": []}

    if extension == ".pdf":
        documents = load_pdf(file_obj, source_name, warnings, report)
    elif extension == ".pptx":
        documents = load_pptx(file_obj, source_name, warnings)
    elif extension in {".xlsx", ".xlsm"}:
        frames = _read_excel_frames(file_obj)
        documents = []
        for sheet_name, dataframe in frames:
            documents.extend(_tabular_chunk_documents(dataframe, source_name, extension.lstrip("."), sheet_name))
            assets.append(_tabular_asset(dataframe, source_name, sheet_name, extension))
    elif extension in {".csv", ".tsv"}:
        dataframe = _read_delimited_frame(file_obj, extension)
        documents = _tabular_chunk_documents(dataframe, source_name, extension.lstrip("."), "Data")
        assets.append(_tabular_asset(dataframe, source_name, "Data", extension))
    elif extension in {".txt", ".md"}:
        documents = load_text_document(file_obj, source_name, extension)
    elif extension == ".json":
        documents = load_json_document(file_obj, source_name)
    elif extension in IMAGE_EXTENSIONS:
        documents = load_image_ocr(file_obj, source_name)
    else:
        documents = []

    if not documents:
        if extension == ".pdf":
            raise OCRRequiredError(f"OCR required: no readable text was found in '{source_name}'.")
        raise ValueError(f"No readable content was found in '{source_name}'.")

    source_id = hashlib.sha256(source_name.encode("utf-8") + b"\0" + raw).hexdigest()
    for document in documents:
        document.metadata["source_id"] = source_id
    for asset in assets:
        asset["source_id"] = source_id
    report["ocr_engine"] = next((doc.metadata["ocr_engine"] for doc in documents
                                  if "ocr_engine" in doc.metadata), None)
    if extension in IMAGE_EXTENSIONS:
        report.update(page_count=len(documents), text_pages=len(documents),
                      ocr_pages=[int(doc.metadata["page"]) for doc in documents])
    return ParsedFile(documents, assets, warnings, report)


def _tabular_asset(dataframe, source_name, sheet_name, extension):
    return {
        "asset_id": f"{source_name}::{sheet_name}",
        "source": source_name,
        "sheet": sheet_name,
        "file_type": extension.lstrip("."),
        "dataframe": dataframe,
        "columns": [str(column) for column in dataframe.columns],
        "row_count": len(dataframe.index),
    }


def load_uploaded_file(uploaded_file):
    return parse_uploaded_file(uploaded_file).documents


def extract_tabular_assets(uploaded_file):
    source_name = getattr(uploaded_file, "name", "uploaded_file")
    extension = Path(source_name).suffix.lower()

    if extension not in TABULAR_EXTENSIONS:
        return []

    return parse_uploaded_file(uploaded_file).tabular_assets
