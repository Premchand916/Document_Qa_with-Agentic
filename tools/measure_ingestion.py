"""Measure isolated parser time and peak worker RSS on synthetic bounded inputs."""

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
import math
import multiprocessing
from pathlib import Path
import platform
import random
import resource
import shutil
import sys
import time

from PIL import Image, ImageDraw

from ingestion.isolated_parser import (DEFAULT_PARSE_TIMEOUT, MAX_PARSE_INPUT_BYTES, MAX_WORKER_RSS,
                                       MAX_WORKER_RSS_GROWTH, parse_in_isolated_process)
from ingestion.file_loader import (MAX_IMAGE_PIXELS, MAX_OCR_CHARS, MAX_PDF_OCR_PAGES,
                                   MAX_TABLE_ROWS, OCR_PAGE_TIMEOUT)

EXPECTED_FRAGMENTS = {
    "representative.txt": "FINAL_TEXT_FACT_42",
    "representative.csv": "9999",
    "representative.json": "item-1399",
    "scanned.pdf": "ALPHA OCR DEADLINE 30 DAYS",
    "rotated.png": "ROTATION POLICY DEADLINE 30 DAYS",
    "noisy.png": "NOISE POLICY DEADLINE 30 DAYS",
}


def _rss_mib(value: int) -> float:
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(value / divisor, 3)


def _measure_worker(connection, name: str, content: bytes) -> None:
    started = time.perf_counter()
    try:
        parsed = parse_in_isolated_process(name, content)
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        combined = "\n".join(document.page_content for document in parsed.documents)
        result = {"status": "passed", "seconds": round(time.perf_counter() - started, 3),
                  "peak_worker_rss_mib": _rss_mib(usage.ru_maxrss),
                  "documents": len(parsed.documents), "tables": len(parsed.tabular_assets),
                  "warnings": parsed.warnings, "ocr_engine": parsed.extraction_report.get("ocr_engine"),
                  "expected_fragment": EXPECTED_FRAGMENTS[name],
                  "expected_fragment_found": EXPECTED_FRAGMENTS[name].casefold() in combined.casefold()}
    except Exception as exc:
        result = {"status": "failed", "seconds": round(time.perf_counter() - started, 3),
                  "error_type": type(exc).__name__, "error": str(exc)[:240]}
    connection.send(result)
    connection.close()


def _embedding_worker(connection, document_count: int) -> None:
    started = time.perf_counter()
    try:
        from embeddings.Hugging_face_embedding import _embedding_batch_size, get_embedding_model

        model = get_embedding_model()
        texts = [f"Policy evidence document {index}: alpha deadline is 30 days. " * 4
                 for index in range(document_count)]
        vectors = model.embed_documents(texts)
        valid = (len(vectors) == document_count and bool(vectors) and bool(vectors[0]) and
                 all(math.isfinite(value) for vector in vectors for value in vector))
        result = {"status": "passed" if valid else "failed",
                  "seconds": round(time.perf_counter() - started, 3),
                  "peak_process_rss_mib": _rss_mib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  "documents": len(vectors), "dimension": len(vectors[0]) if vectors else 0,
                  "batch_size": _embedding_batch_size()}
    except Exception as exc:
        result = {"status": "failed", "seconds": round(time.perf_counter() - started, 3),
                  "error_type": type(exc).__name__, "error": str(exc)[:240]}
    connection.send(result)
    connection.close()


def _scanned_pdf() -> bytes:
    image = Image.new("RGB", (1800, 500), "white")
    ImageDraw.Draw(image).text((80, 180), "ALPHA  OCR  DEADLINE  30  DAYS",
                               fill="black", font_size=72)
    output = BytesIO()
    image.save(output, format="PDF", resolution=150)
    return output.getvalue()


def _rotated_image() -> bytes:
    image = Image.new("RGB", (1600, 1200), "white")
    drawing = ImageDraw.Draw(image)
    for index in range(7):
        drawing.text((80, 80 + index * 145), "ROTATION POLICY DEADLINE 30 DAYS",
                     fill="black", font_size=52)
    output = BytesIO()
    image.rotate(90, expand=True, fillcolor="white").save(output, format="PNG")
    return output.getvalue()


def _noisy_image() -> bytes:
    image = Image.new("L", (1800, 500), 255)
    ImageDraw.Draw(image).text((80, 180), "NOISE POLICY DEADLINE 30 DAYS", fill=0, font_size=72)
    generator = random.Random(42)
    for _ in range(15_000):
        image.putpixel((generator.randrange(image.width), generator.randrange(image.height)),
                       0 if generator.random() < 0.5 else 255)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _cases() -> dict[str, bytes]:
    text = (("alpha policy evidence " * 32) + "\n") * 1024 + "FINAL_TEXT_FACT_42"
    csv = "amount,region\n" + "".join(f"{index},North\n" for index in range(10_000))
    structured = {f"item-{index}": {"value": index, "status": "active"} for index in range(1_400)}
    return {"representative.txt": text.encode(), "representative.csv": csv.encode(),
            "representative.json": json.dumps(structured).encode(), "scanned.pdf": _scanned_pdf(),
            "rotated.png": _rotated_image(), "noisy.png": _noisy_image()}


def _start_measurement(context, name: str, content: bytes):
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_measure_worker, args=(child, name, content))
    process.start()
    child.close()
    return parent, process


def _receive_measurement(parent, process, timeout: float) -> dict:
    try:
        if parent.poll(timeout):
            try:
                return parent.recv()
            except (EOFError, OSError) as exc:
                return {"status": "failed", "error_type": "MeasurementProcessError",
                        "error": str(exc)[:240]}
        process.terminate()
        return {"status": "failed", "error_type": "MeasurementTimeout"}
    finally:
        parent.close()
        process.join(timeout=3)
        if process.is_alive():
            process.kill()
            process.join(timeout=3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--embedding-documents", type=int, default=256)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 2:
        parser.error("--concurrency must match the API parser bound: 1 or 2")
    if not 1 <= args.embedding_documents <= 2_000:
        parser.error("--embedding-documents must be between 1 and 2000")
    context = multiprocessing.get_context("spawn")
    cases = _cases()
    results = {}
    for name, content in cases.items():
        parent, process = _start_measurement(context, name, content)
        results[name] = _receive_measurement(parent, process, DEFAULT_PARSE_TIMEOUT + 10)

    concurrent_names = ["scanned.pdf", "rotated.png"][:args.concurrency]
    concurrent_started = time.perf_counter()
    running = [(name, *_start_measurement(context, name, cases[name])) for name in concurrent_names]
    concurrent_results = {
        name: _receive_measurement(parent, process, DEFAULT_PARSE_TIMEOUT + 10)
        for name, parent, process in running
    }
    concurrent = {
        "workers": args.concurrency,
        "seconds": round(time.perf_counter() - concurrent_started, 3),
        "cases": concurrent_results,
        "peak_worker_rss_upper_bound_mib": round(sum(
            case.get("peak_worker_rss_mib", 0) for case in concurrent_results.values()
        ), 3),
    }

    embedding_parent, embedding_child = context.Pipe(duplex=False)
    embedding_process = context.Process(target=_embedding_worker,
                                        args=(embedding_child, args.embedding_documents))
    embedding_process.start()
    embedding_child.close()
    embedding = _receive_measurement(embedding_parent, embedding_process, 120)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "python": platform.python_version(), "tesseract_available": bool(shutil.which("tesseract"))},
        "configured_limits": {"parse_timeout_seconds": DEFAULT_PARSE_TIMEOUT,
                              "input_bytes": MAX_PARSE_INPUT_BYTES, "table_rows": MAX_TABLE_ROWS,
                              "worker_rss_growth_bytes": MAX_WORKER_RSS_GROWTH,
                              "worker_rss_emergency_bytes": MAX_WORKER_RSS,
                              "image_pixels": MAX_IMAGE_PIXELS, "pdf_ocr_pages": MAX_PDF_OCR_PAGES,
                              "ocr_page_timeout_seconds": OCR_PAGE_TIMEOUT, "ocr_characters": MAX_OCR_CHARS},
        "cases": results,
        "concurrent_parser_measurement": concurrent,
        "embedding_measurement": embedding,
        "interpretation": ("Synthetic current-machine parser and cached-embedding measurement. Concurrent RSS is a "
                           "conservative sum of per-worker peaks, not a synchronized process-tree sample or production SLO."),
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)
    parser_passed = all(case["status"] == "passed" and case["expected_fragment_found"]
                        for case in results.values())
    concurrent_passed = all(case["status"] == "passed" and case["expected_fragment_found"]
                            for case in concurrent_results.values())
    return 0 if parser_passed and concurrent_passed and embedding["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
