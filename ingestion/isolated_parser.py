"""Run untrusted document parsing in a short-lived process with a hard deadline."""

from io import BytesIO, StringIO
import json
import multiprocessing
import sys

import pandas as pd
from langchain_core.documents import Document

from ingestion.file_loader import IngestionLimitError, OCRRequiredError, ParsedFile, parse_uploaded_file

DEFAULT_PARSE_TIMEOUT = 30.0
MAX_RESULT_BYTES = 64 * 1024 * 1024
MAX_WORKER_ADDRESS_SPACE = 2 * 1024 * 1024 * 1024
MAX_PARSE_INPUT_BYTES = 10 * 1024 * 1024


class ParserTimeoutError(Exception):
    """The parser exceeded its wall-clock deadline and was terminated."""


class ParserUnavailableError(Exception):
    """The parser process exited without a valid result."""


def _limit_worker_memory():
    # RLIMIT_AS is reliable for Linux worker processes. macOS uses the hard
    # process deadline; its virtual-memory accounting is less predictable.
    if sys.platform == "linux":
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (MAX_WORKER_ADDRESS_SPACE, MAX_WORKER_ADDRESS_SPACE))


def _encode_result(parsed: ParsedFile) -> dict:
    assets = []
    for asset in parsed.tabular_assets:
        metadata = {key: value for key, value in asset.items() if key != "dataframe"}
        metadata["table"] = json.loads(asset["dataframe"].to_json(orient="table", index=False, date_format="iso"))
        assets.append(metadata)
    payload = {"documents": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in parsed.documents],
               "tabular_assets": assets, "warnings": parsed.warnings,
               "extraction_report": parsed.extraction_report}
    return payload


def _decode_result(payload: dict) -> ParsedFile:
    documents = [Document(page_content=row["page_content"], metadata=row["metadata"]) for row in payload["documents"]]
    assets = []
    for row in payload["tabular_assets"]:
        asset = {key: value for key, value in row.items() if key != "table"}
        asset["dataframe"] = pd.read_json(StringIO(json.dumps(row["table"])), orient="table")
        assets.append(asset)
    return ParsedFile(documents, assets, payload["warnings"], payload["extraction_report"])


def _worker(connection, name: str, content: bytes):
    try:
        _limit_worker_memory()
        file_obj = BytesIO(content)
        file_obj.name = name
        response = {"status": "ok", "result": _encode_result(parse_uploaded_file(file_obj))}
    except IngestionLimitError:
        response = {"status": "limit"}
    except OCRRequiredError:
        response = {"status": "ocr_required"}
    except Exception:
        response = {"status": "invalid"}
    try:
        encoded = json.dumps(response, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_RESULT_BYTES:
            encoded = b'{"status":"limit"}'
        connection.send_bytes(encoded)
    finally:
        connection.close()


def parse_in_isolated_process(name: str, content: bytes, *, timeout: float = DEFAULT_PARSE_TIMEOUT) -> ParsedFile:
    if timeout <= 0:
        raise ValueError("Parser timeout must be positive")
    if len(content) > MAX_PARSE_INPUT_BYTES:
        raise IngestionLimitError(f"Parser input exceeds {MAX_PARSE_INPUT_BYTES} bytes")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(child, name, content), daemon=True)
    started = False
    try:
        process.start()
        started = True
        child.close()
        if not parent.poll(timeout):
            raise ParserTimeoutError()
        response = json.loads(parent.recv_bytes(MAX_RESULT_BYTES))
        status = response.get("status")
        if status == "limit":
            raise IngestionLimitError("Parser resource limit exceeded")
        if status == "ocr_required":
            raise OCRRequiredError("PDF requires OCR")
        if status != "ok":
            raise ValueError("Document parser could not read the file")
        return _decode_result(response["result"])
    except (EOFError, OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ParserUnavailableError("Parser process failed") from exc
    finally:
        parent.close()
        child.close()
        if started and process.is_alive():
            process.terminate()
        if started:
            process.join(timeout=2)
        if started and process.is_alive():
            process.kill()
            process.join(timeout=2)
