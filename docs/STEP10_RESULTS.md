# Step 10 results — bounded execution and operational visibility

Verified 28 October 2026. Local deterministic foundation gate passes; production load and multi-host readiness are not claimed.

## Implemented

- Every upload is persisted before processing as an owner-scoped SQLite job with bounded input bytes, payload fingerprint, idempotency key, expected corpus version, status, stage timing, and expiry.
- Job states are `queued`, `running`, `cancel_requested`, `ready`, `failed`, and `cancelled`. Opaque expiring leases prevent two shared-local workers from owning one job. Startup and status polling recover expired work without stealing live leases.
- Parse, chunk, embedding/index construction, question generation, and publication execute outside the event loop under a two-job semaphore. Workspace pending jobs default to ten.
- Existing `/api/upload` synchronous success behavior remains supported. `Prefer: respond-async` returns HTTP 202; owner-scoped status/list/cancel APIs support polling and cooperative cancellation.
- UI uses async upload, displays durable stage state, resumes active jobs after reload, and exposes cancellation.
- Cancellation is checked between stages. Parser deadlines remain process-enforced. Capacity stays reserved when a client disconnects or chat reaches its response deadline while native work continues.
- Corpus publication uses the job ID as its fixed version. Retrying after a committed-but-lost response verifies and reuses that version instead of publishing a duplicate.
- Failures store safe codes and messages. Ingestion capacity, parser deadline, invalid input, provider unavailable, chat capacity, and chat deadline are distinguishable without exposing internal exception text.
- Terminal jobs and uploaded input bytes default to seven-day metadata retention; input BLOBs are removed immediately at terminal completion.

## Verification

- Full offline baseline passes 159 tests. Historical diagnostics report zero reproduced defects, 25 not reproduced, one passing rejection control, and zero probe errors.
- Real local text/CSV/JSON ingestion, PDF/image OCR, concurrent parser measurement, cached embedding measurement, and environment checks pass. Cached embedding measurement processed 256 documents at 384 dimensions in 2.543 seconds with 595.125 MiB peak process RSS.
- Duplicate identical upload creates one job and corpus version; mismatched idempotency reuse returns 409.
- Expired leases recover; live leases are not stolen; stale workers cannot complete a job.
- Restart completes an expired job from persisted input.
- Cooperative cancellation before publication keeps the previous corpus active.
- Synthetic indexing failure returns a safe error and preserves the previous corpus.
- Twenty lightweight state reads during blocked parsing keep synthetic local p95 below the provisional 250 ms target.
- Chat deadline returns 504, provider failure returns 503, and timed-out native work retains its semaphore slot until exit.
- Owner/workspace isolation, bounded two-job execution, stage timing, terminal cleanup, and committed-response-loss replay have deterministic tests.
- Live browser check received HTTP 202, polled the durable job endpoint, and transitioned to the ready UI with the uploaded source and suggested question.

## Remaining limits

- SQLite jobs and in-process semaphores support one host with shared local storage. Production multi-host execution requires a shared database/queue, database time, worker identity, and distributed admission control.
- Cancellation is cooperative between stages. Active native embedding/model work cannot be safely killed; isolated subprocess execution is still limited to parsing.
- Status polling drives recovery after startup. A dedicated scheduler/worker deployment is preferable for unattended queued work.
- Local synthetic p95 is not a production SLO. Browser heap, real Ollama/Gemini latency, token use, peak full-process RSS, four-client load, and long-corpus behavior remain unmeasured.
- FastAPI startup hooks and current dependency deprecation warnings remain modernization work.

Next: Step 11 held-out answer/evidence evaluation, migration rollback rehearsal, and Flat-versus-ANN measurements at demonstrated scale.
