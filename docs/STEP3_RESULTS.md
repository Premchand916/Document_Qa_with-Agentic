# Step 3 — Ingestion progress

Verified 27 October 2026. Step 3 is **in progress**; this is an implementation slice, not its acceptance gate.

## Changed behavior

- FastAPI calls one parser per uploaded file. CSV/Excel frames supply both embedding text and typed table assets, avoiding the previous duplicate read.
- Missing numeric CSV/Excel cells retain nullable numeric dtypes. The `amount` fixture `[10, null, 30]` totals **40** and remains `Int64` in the tested Pandas environment.
- JSON, slides, table rows, and ordinary text all pass through the same chunk bound. The configured encoder tokenizer is used when present; an offline UTF-8 byte bound is the fallback. A 240-token content budget reserves room for special tokens under the current MiniLM 256-token input limit. Each chunk carries stable `source_id`, `chunk_id`, and `chunk_index` metadata alongside page/sheet/row references. IDs derive from source bytes/name and chunk content/position; the corpus version is still assigned by publication.
- Office archives reject more than 500 entries or 100 MiB declared expansion. PDFs reject more than 500 pages, presentations more than 500 slides, and parsed tables more than 100,000 rows. Empty-text PDFs return `ocr_required`; partially unreadable PDFs/slides return warnings in the upload response and UI.
- Upload parsing now runs in a short-lived child process with a 30-second wall deadline and at most two concurrent parser jobs per API process. The parent terminates timed-out workers, bounds the serialized result to 64 MiB, and leaves the active corpus unchanged on failure. Linux workers also set a 2 GiB address-space limit. API timeout and unavailable responses use stable error codes.
- JSON scalar values and empty containers now retain RFC 6901 pointer metadata through chunking and corpus storage. Structure traversal stops after 5,000 nodes or 64 levels. Duplicate values at distinct paths receive distinct chunk IDs.
- PDF text is grouped into blocks of up to eight lines. Blocks retain page, line range, top-origin bounding box, and page size; extraction stops after 20,000 lines. Fallback page text marks coordinates unavailable. Partial PDFs name pages needing OCR. Each uploaded source has a structured extraction report in the upload response and persisted `/api/state`.

## Evidence

The original Step 3 slice passed **79 tests** and reduced reproduced limitations to **15**. Follow-up work capped CSV/Excel reads at the row limit plus one row and persisted extraction warnings in `corpus.json`; `/api/state` now returns them after reload. The current offline suite passes **105 tests**; the diagnostic harness reports **9 reproduced limitations** and no probe errors. Isolated CSV and PDF parsing preserve typed tables, PDF coordinates, and source reports across the process boundary. A forced deadline ends the child; an API timeout preserves the previous corpus version. A live HTTP upload succeeded through the production parser path. Upload/publish/reload tests retain the last fact in long JSON with its `/body` pointer, plus PDF bounding boxes and reports in stored FAISS documents and corpus state. This is deterministic evidence, not a model-quality measurement.

## Remaining Step 3 work

- PDF coordinates identify text blocks, not exact positions of every generated token chunk. Image/slide OCR quality and fully scanned PDF handling still need evaluation; a fully scanned PDF requires OCR before upload.
- macOS does not have a verified hard worker-memory cap. Linux's address-space limit has not been exercised in this environment. Library decompression, image/OCR work, total request memory, and production concurrency remain unmeasured. The child deadline bounds parser lifetime; it does not bound every parent-side allocation or embedding job.
- Warning messages now survive reload with their corpus version. Legacy Streamlit still calls compatibility wrappers separately.
- The tokenizer budget is based on current MiniLM settings. Step 4 persists and validates encoder/tokenizer identity for new corpora, including local real-model checks.
- Chrome synthetic preview confirmed initial state, suggested-question chat with source evidence, replacement upload, and a partially blank PDF warning in the live interface. A follow-up browser check confirmed the warning remains visible after page reload. Mobile layout remains unverified in this slice; the known narrow-viewport defect from Step 2 remains open.

Next: measure parser resource budgets with representative files and evaluate OCR on scanned documents. Step 6 durable conversations can proceed from the completed workspace foundation. Sign-in and connectors follow the product expansion sequence in the [implementation plan](IMPLEMENTATION_PLAN.md).
