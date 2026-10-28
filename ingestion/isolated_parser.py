"""Run untrusted document parsing in a short-lived process with a hard deadline."""

from io import BytesIO, StringIO
import json
import multiprocessing
import sys
import time

import pandas as pd
import psutil
from langchain_core.documents import Document

from ingestion.file_loader import (IngestionLimitError, OCRConfigurationError, OCRRequiredError,
                                   ParsedFile, parse_uploaded_file)

DEFAULT_PARSE_TIMEOUT = 30.0
MAX_RESULT_BYTES = 64 * 1024 * 1024
MAX_WORKER_ADDRESS_SPACE = 2 * 1024 * 1024 * 1024
MAX_WORKER_RSS_GROWTH = 512 * 1024 * 1024
MAX_WORKER_RSS = 1536 * 1024 * 1024
MAX_PARSE_INPUT_BYTES = 10 * 1024 * 1024
RESOURCE_POLL_INTERVAL = 0.05


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


def _worker_resident_bytes(process_id: int) -> int:
    """Return worker plus descendant RSS; missing short-lived processes count as zero."""
    try:
        process = psutil.Process(process_id)
        resident = process.memory_info().rss
        try:
            descendants = process.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
            return resident
        for child in descendants:
            try:
                resident += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                continue
        return resident
    except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
        return 0


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


def _send_response(connection, response: dict) -> None:
    encoded = json.dumps(response, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_RESULT_BYTES:
        encoded = b'{"status":"limit"}'
    connection.send_bytes(encoded)


def _worker(connection, name: str, content: bytes):
    try:
        _limit_worker_memory()
        _send_response(connection, {"status": "ready", "baseline_rss": psutil.Process().memory_info().rss})
        file_obj = BytesIO(content)
        file_obj.name = name
        response = {"status": "ok", "result": _encode_result(parse_uploaded_file(file_obj))}
    except IngestionLimitError:
        response = {"status": "limit"}
    except OCRRequiredError:
        response = {"status": "ocr_required"}
    except OCRConfigurationError:
        response = {"status": "ocr_configuration"}
    except Exception:
        response = {"status": "invalid"}
    try:
        _send_response(connection, response)
    finally:
        connection.close()


def _receive_response(parent, process, deadline: float, rss_limit: int) -> dict:
    while True:
        if _worker_resident_bytes(process.pid) > rss_limit:
            raise IngestionLimitError("Parser resident-memory limit exceeded")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ParserTimeoutError()
        if parent.poll(min(RESOURCE_POLL_INTERVAL, remaining)):
            return json.loads(parent.recv_bytes(MAX_RESULT_BYTES))
        if not process.is_alive():
            raise ParserUnavailableError("Parser process exited before returning a result")


def _worker_rss_limit(baseline_rss: int) -> int:
    return min(MAX_WORKER_RSS, max(0, baseline_rss) + MAX_WORKER_RSS_GROWTH)


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
        deadline = time.monotonic() + timeout
        response = _receive_response(parent, process, deadline, MAX_WORKER_RSS)
        if response.get("status") == "ready":
            baseline_rss = max(0, int(response.get("baseline_rss", 0)))
            response = _receive_response(parent, process, deadline, _worker_rss_limit(baseline_rss))
        status = response.get("status")
        if status == "limit":
            raise IngestionLimitError("Parser resource limit exceeded")
        if status == "ocr_required":
            raise OCRRequiredError("PDF requires OCR")
        if status == "ocr_configuration":
            raise OCRConfigurationError("Local OCR configuration is unavailable")
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
