"""Measure isolated parser time and peak worker RSS on synthetic bounded inputs."""

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
import multiprocessing
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

from PIL import Image, ImageDraw

from ingestion.isolated_parser import DEFAULT_PARSE_TIMEOUT, MAX_PARSE_INPUT_BYTES, parse_in_isolated_process
from ingestion.file_loader import (MAX_IMAGE_PIXELS, MAX_OCR_CHARS, MAX_PDF_OCR_PAGES,
                                   MAX_TABLE_ROWS, OCR_PAGE_TIMEOUT)

EXPECTED_FRAGMENTS = {
    "representative.txt": "FINAL_TEXT_FACT_42",
    "representative.csv": "9999",
    "representative.json": "item-1399",
    "scanned.pdf": "ALPHA OCR DEADLINE 30 DAYS",
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


def _scanned_pdf() -> bytes:
    image = Image.new("RGB", (1800, 500), "white")
    ImageDraw.Draw(image).text((80, 180), "ALPHA  OCR  DEADLINE  30  DAYS",
                               fill="black", font_size=72)
    output = BytesIO()
    image.save(output, format="PDF", resolution=150)
    return output.getvalue()


def _cases() -> dict[str, bytes]:
    text = (("alpha policy evidence " * 32) + "\n") * 1024 + "FINAL_TEXT_FACT_42"
    csv = "amount,region\n" + "".join(f"{index},North\n" for index in range(10_000))
    structured = {f"item-{index}": {"value": index, "status": "active"} for index in range(1_400)}
    return {"representative.txt": text.encode(), "representative.csv": csv.encode(),
            "representative.json": json.dumps(structured).encode(), "scanned.pdf": _scanned_pdf()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    context = multiprocessing.get_context("spawn")
    results = {}
    for name, content in _cases().items():
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=_measure_worker, args=(child, name, content))
        process.start()
        child.close()
        if parent.poll(DEFAULT_PARSE_TIMEOUT + 10):
            results[name] = parent.recv()
        else:
            results[name] = {"status": "failed", "error_type": "MeasurementTimeout"}
            process.terminate()
        parent.close()
        process.join(timeout=3)
        if process.is_alive():
            process.kill()
            process.join(timeout=3)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "python": platform.python_version(), "tesseract_available": bool(shutil.which("tesseract"))},
        "configured_limits": {"parse_timeout_seconds": DEFAULT_PARSE_TIMEOUT,
                              "input_bytes": MAX_PARSE_INPUT_BYTES, "table_rows": MAX_TABLE_ROWS,
                              "image_pixels": MAX_IMAGE_PIXELS, "pdf_ocr_pages": MAX_PDF_OCR_PAGES,
                              "ocr_page_timeout_seconds": OCR_PAGE_TIMEOUT, "ocr_characters": MAX_OCR_CHARS},
        "cases": results,
        "interpretation": "Synthetic parser-process measurement; excludes embedding/model work and production concurrency.",
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)
    return 0 if all(case["status"] == "passed" and case["expected_fragment_found"]
                    for case in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
