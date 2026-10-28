# Foundation engineering knowledge base

Use this record for foundation changes: understand requirements, state invariants, implement the smallest coherent slice, validate failure paths, and record remaining limits. Do not claim production readiness from local tests.

## Required workflow

1. Inspect repository guidance, current implementation, tests, and dirty state.
2. Convert limitations into explicit requirements and acceptance cases.
3. Choose one authoritative owner for each kind of state.
4. Add migration-safe code and deterministic tests before broad validation.
5. Run focused tests, full offline suite, failure diagnostics, and relevant real-resource checks.
6. Update the implementation plan and dated result document with verified evidence and remaining limits.

## Active work — Step 6 durable conversation completion

### Requirements

- SQLite remains the only authoritative transcript. LangGraph state stays request-scoped until it becomes fully serializable.
- Multiple API workers must not invalidate another worker's active turn during startup.
- Only the worker holding a live, opaque lease may renew, complete, or fail its turn. Expired or replaced workers must fail closed.
- Duplicate completed requests replay the stored response. Conflicting active requests return 409.
- Conversation deletion must be authorized by workspace and principal and remove turns plus summaries in one transaction.
- Retention must have a documented default, bounded configuration, and deterministic cleanup behavior.
- Prompt history must remain bounded. Older completed turns may contribute through a deterministic recap while originals remain until deletion or retention cleanup.
- Conversation detail must be bounded and cursor-paginated.

### Design

- Add `lease_token` and `lease_expires_at` to turns through additive SQLite migration.
- Use a 60-second lease with a 20-second API heartbeat. Startup recovers expired leases only.
- Require lease token for `renew`, `complete`, and `fail`; stale tokens cannot mutate a reclaimed turn.
- Default retention: 90 days. Allow `DOCQA_CONVERSATION_RETENTION_DAYS=1..3650`.
- Store a bounded extractive recap of turns older than the eight-turn prompt window. No LLM call, no second transcript, no deletion of originals.
- Hard-delete one authorized conversation transactionally. A later deletion/export policy can extend this same repository boundary.
- Return at most 50 completed turns per detail request, with `before_sequence` cursor and `next_before_sequence`.

### Acceptance cases

- Second repository instance sees a live lease and cannot reserve the same conversation.
- Expired lease can be reclaimed; original token cannot complete or fail the replacement.
- Heartbeat renewal preserves ownership through a long graph call.
- Starting another API worker does not fail a live turn.
- Delete removes conversation, turns, and recap only for the authorized principal/workspace.
- Retention removes only conversations older than cutoff and never crosses ownership boundaries.
- Long conversations expose a recap to prompts and paginate detail without returning all rows.
- Full offline baseline and historical failure diagnostics remain green.

### Explicit non-goals

- Model-call hard cancellation, durable background jobs, and process-wide admission control remain Step 10.
- Semantic LLM summarization is deferred until evaluated; deterministic recap is safer foundation behavior.
- Cross-region SQLite deployment is unsupported. Shared workers must use the same local filesystem and SQLite database.

### Result — 28 October 2026

- Implemented additive leases, heartbeat renewal, stale-token rejection, expired-only startup recovery, WAL, deletion, retention, recap, pagination, clarification, UI deletion, and responsive layout.
- Full baseline passed: 144 tests; zero reproduced defects; 25 defects not reproduced; one rejection control passed; zero probe errors; ingestion, embedding, and environment stages passed.
- Live synthetic browser check passed conversation creation, delete confirmation, transactional API deletion, and empty-state reset.
- Remaining boundaries: shared local SQLite only, startup-triggered retention, deterministic recap, request-scoped graph objects, and no hard model-call cancellation.

## Active work — Step 9 selective episodic feedback

### Requirements

- Saving feedback is explicit and opt-in; ordinary chat turns never become episodes automatically.
- Public clients may submit corrections but cannot mark them verified.
- Episode fields come from the authorized completed turn: task, result, tool output, corpus version, citations, and sources.
- Recall requires verified status, unexpired timestamp, exact workspace/principal, normalized task key, and current corpus version.
- Episodes guide the answer process but never count as evidence or citations.
- Rejected, stale, deleted, unverified, cross-principal, and cross-workspace episodes are excluded.
- Conversation deletion and retention must remove linked episodes through the same SQLite transaction boundary.

### Design

- Store episodes in the existing conversation SQLite database with a composite foreign key and `ON DELETE CASCADE`.
- Public API creates `unverified` feedback and supports owner-scoped list/delete. Verification remains an internal repository operation until an administrator/validator role exists.
- Use deterministic normalized task keys before considering a separate embedding index.
- Default episode expiry: 180 days, configurable from 1–3650 days.
- Inject bounded verified correction text into prompt guidance with an explicit “not evidence” label.

### Acceptance cases

- Verified correction is recalled for the same task/corpus and appears in prompt guidance.
- Unverified/rejected/expired/cross-owner/cross-workspace/wrong-corpus episodes are absent.
- Client payload cannot set validation status, result, sources, or evidence versions.
- Deleting a conversation removes its episodes; deleting one episode does not delete transcript data.

### Result — 28 October 2026

- Implemented opt-in feedback capture, server-derived provenance, internal-only verification, bounded exact-task recall, corpus isolation, expiry, owner-scoped APIs, and cascade deletion.
- Full baseline passed: 148 tests; zero reproduced defects; 25 defects not reproduced; one rejection control passed; zero probe errors; ingestion, embedding, and environment stages passed.
- Live synthetic browser check passed correction entry, unverified status messaging, and successful `POST /api/episodes` persistence.
- Remaining boundaries: no trusted reviewer UI or audit role, conservative exact-task/corpus recall, startup-triggered cleanup, and no evaluated semantic episode index.

## Active work — Step 10 durable ingestion jobs

### Requirements

- Preserve the existing synchronous `/api/upload` success response while backing it with a durable job record.
- Derive workspace and principal from the trusted request boundary; never accept ownership in multipart fields.
- Persist bounded input bytes before work starts. One idempotency key plus one payload fingerprint identifies one job; conflicting reuse fails closed.
- Run parse, chunk, embedding/index construction, question generation, and publication outside the event loop under bounded admission.
- Use opaque expiring leases so multiple local workers cannot process one job concurrently. Restart requeues only expired work.
- Keep the previous corpus active until the new version is fully written, verified, and atomically published.
- Expose owner-scoped job status, safe error codes, queue/run duration, stage timings, and cooperative cancellation.
- Client disconnect must not release capacity while blocking native work continues.

### Design

- Add a SQLite ingestion-job repository beside workspace storage. Store bounded input BLOBs, state, lease, expected corpus version, result/error, and timing metadata.
- Use states `queued`, `running`, `cancel_requested`, `ready`, `failed`, and `cancelled`.
- Keep a bounded in-process task registry and semaphore. SQLite leases provide shared-local-worker exclusion; PostgreSQL or another shared database remains required for multi-host execution.
- Check cancellation between durable stages. Parser hard deadlines remain process-enforced; native embedding/model calls are not forcibly interrupted.
- Retain jobs for seven days by default, configurable within bounded limits; startup purges expired terminal jobs and recovers expired leases.

### Acceptance cases

- Duplicate identical submission returns one job/result; conflicting idempotency reuse returns 409.
- Slow ingestion does not stall lightweight API requests.
- Expired running work is recovered after restart; live work is not stolen.
- Cancellation before publication preserves the previous corpus.
- Failure stores a safe code and never reports success or exposes internal exception text.
- Cross-workspace/principal job reads and cancellation fail closed.
- Stage and queue timing are visible without exposing document contents.

### Explicit non-goals

- Hard cancellation of an active native embedding or model call.
- Multi-host scheduling, distributed queues, and globally fair admission.
- Browser heap and production p95 claims from synthetic local tests.

### Result — 28 October 2026

- Implemented durable owner-scoped upload jobs, input persistence, bounded admission, expiring leases, restart/poll recovery, replay-safe publication, stage metrics, safe failures, async UI polling, and cooperative cancellation.
- Full baseline passed: 159 tests; zero reproduced defects; 25 defects not reproduced; one rejection control passed; zero probe errors; ingestion, OCR, embedding, and environment stages passed.
- Cached embedding measurement passed for 256 documents at 384 dimensions in 2.543 seconds with 595.125 MiB peak process RSS. This is a local synthetic measurement, not a production SLO.
- Live browser check passed HTTP 202 upload acceptance, durable status polling, final source display, suggested question display, and ready-state transition.
- Remaining boundaries: shared-local SQLite only, polling/startup recovery instead of a dedicated worker scheduler, cooperative native-call cancellation, and unmeasured production/model/browser load.

## Active work — Step 11 evaluation, migration, and index decisions

### Requirements

- Keep development labels separate from held-out release evidence. Never claim model quality from deterministic tests or visible tuning fixtures.
- Score ordered evidence Recall@k and MRR, source-level citation precision/recall, literal answer contracts, abstention, exact table outputs, latency, tokens, cost, and optional manual semantic review.
- Reject incomplete, duplicate, unknown, malformed, or split-mismatched prediction runs. Release eligibility requires an explicitly held-out label set, full coverage, declared model/run metadata, and passing targets.
- Rehearse corpus migration and rollback only on a copy. Verify both previous and current versions after process-style repository reconstruction; never mutate the source data directory.
- Keep exact Flat search as reference. Benchmark HNSW against exact neighbors using deterministic vectors and report build/search time, serialized size, and Recall@k.
- Do not change the production index from Flat unless exact search misses a declared latency target and ANN meets declared recall plus improvement targets.

### Design

- Extend the offline evaluation tool while preserving legacy prediction-list input for development scoring. Add a versioned run-envelope for release evaluation.
- Add reusable release targets to label files; current synthetic fixture remains `development` and therefore cannot become release eligible.
- Add a workspace version-list API and a copy-only migration rehearsal utility using SQLite backup plus immutable corpus copies.
- Add an offline FAISS benchmark with fixed seed, bounded vector/query counts, exact Flat baseline, and HNSW parameter reporting. Benchmark output recommends; it does not mutate runtime configuration.

### Acceptance cases

- Ranking metrics change correctly when relevant sources move down or disappear.
- Citation hallucinations reduce source precision; missing expected citations reduce source recall.
- Table value/type mismatch fails exact table accuracy; correct abstention is scored separately from answer fragments.
- Missing run metadata or development labels make release eligibility false even when answers pass.
- Rollback rehearsal loads old data, restores current data, survives repository reconstruction, and leaves source state/checksums unchanged.
- Flat returns exact reference neighbors; HNSW Recall@k is measured against them; recommendation remains Flat when exact latency target passes.

### Explicit non-goals

- Claiming held-out real-model quality before a separately collected evaluation run exists.
- Switching production storage to HNSW, IVF, PQ, LSH, or another backend from synthetic microbenchmarks alone.
- Online migration while writers are active or cross-host rollback orchestration.

### Result — 28 October 2026

- Implemented versioned evaluation/run contracts, ranked retrieval and citation metrics, typed table scoring, abstention cleanliness, latency/token/cost accounting, manual-review gates, and combined release-evidence verification.
- Implemented copy-only SQLite/artifact migration rehearsal with rollback, restoration, repository reconstruction, and source immutability checks.
- Added isolated Flat/HNSW benchmark to the full baseline. At 5,000×64D vectors, exact Flat p95 was 0.0203 ms; HNSW p95 was 0.0395 ms with Recall@10 0.981 and 2.06× serialized size. Decision: retain Flat.
- Full baseline passed: 169 tests; zero reproduced defects; 25 defects not reproduced; one rejection control passed; zero probe errors; ingestion, OCR, embeddings, index benchmark, and environment stages passed.
- Release/fine-tuning gate remains blocked by design: current labels are development data and no separately frozen held-out real-model run plus independent review exists.

## Active work — Step 12 label-blind candidate evidence

### Requirements

- Inference code must never load labels, rubrics, expected fragments, release targets, or review decisions.
- Frozen tasks and source bytes must hash to the identity recorded by labels and the prediction run.
- Record actual model and embedding revisions, graph latency, model-call tokens, retrieved/cited sources, abstention, and exact tool results.
- A dataset stops being held out as soon as its result influences implementation or labels. Machine-mark exposed sets non-release.
- Release eligibility requires a fresh release-candidate set, all automated targets, and complete independent human review metadata and decisions.

### Design

- Keep `tasks.json` and `labels.json` separate. Recursively reject label-only keys from the task manifest.
- Capture aggregate-safe model metadata through a request-local context variable; never persist prompts or response content in telemetry.
- Use a manual-review worksheet with immutable candidate identity. Attach only complete decisions from a named independent reviewer.
- When the optional cross-encoder is absent, combine relative BM25 filtering with a calibrated embedding-distance rejection. Surface the chosen relevance-gate path.
- Resolve explicit version references deterministically before generation.

### Result — 28 October 2026

- Built and tested the complete label-blind Ollama evaluation, review, scoring, migration, and combined-gate workflow.
- The calibration run passed automated targets but is permanently non-release because it drove fixes.
- A fresh release attempt failed citation precision at 0.786 versus 0.80 and exposed wrong-version follow-up retrieval. The set was retired before fixes.
- Added relative lexical filtering, exact-version retrieval, clean abstention, and evidence-review preservation from those observations.
- Release and fine-tuning remain blocked. Next evidence must come from a new post-fix set and an independent human reviewer.

## Active work — Step 13 four-layer memory

### Requirements

- Keep working, semantic, episodic, and procedural memory separate; each has different authority, trust, recall, retention, and deletion semantics.
- Never auto-extract semantic facts or procedures from ordinary chat.
- Semantic memory must be explicit, owner/workspace scoped, bounded, expiring, inspectable, replaceable, and deletable.
- Semantic and episodic context guide responses but never count as document or web evidence.
- Procedures remain code/operator-owned until trusted administrator approval and audit roles exist.

### Design

- Continue using `ConversationRepository` for bounded working history and recap.
- Store semantic memory in the shared local conversation SQLite database without a conversation foreign key, so conversation deletion does not erase separately approved stable preferences.
- Keep episodes tied to source conversations and corpus versions with verified-only recall.
- Treat graph routing, allowlisted tools, prompt library, and operator-selected prompt skills as procedural memory. Record prompt-skill hash/source; do not discover arbitrary home-directory files.

### Result — 28 October 2026

- Added semantic-memory repository, APIs, UI controls, expiry, capacity/character bounds, owner isolation, and prompt integration.
- Added persistent secret and instruction-override rejection.
- Hardened procedural guidance discovery to repository or explicit operator configuration and added SHA-256 identity.
- Added frozen deterministic memory evaluation for recall, conflict replacement, isolation, expiry, deletion, injection rejection, evidence separation, and latency.
- Full baseline passed 187 tests; zero reproduced diagnostic defects; 25 defects not reproduced; one rejection control passed; zero failed controls or probe errors; ingestion, index, memory, and environment stages passed.
- Local SQLite recall was 0.441 ms p95 across 100 iterations. This is development evidence, not a production SLO or model-quality measurement.
- The deterministic memory foundation is complete. Fine-tuning remains paused; any next memory-quality run must be label-blind and model-backed.

## Active work — Step 14 memory limitation hardening

### Requirements

- Improve semantic selection without adding model/network dependency or weakening hard prompt budgets.
- Give users an owner-scoped semantic-memory export.
- Make episodic review decisions auditable while keeping verification outside public user APIs.
- Provide one bounded cleanup target for deployment schedulers.
- Restore memory controls at tablet widths.

### Result — 28 October 2026

- Added deterministic query/key/value overlap ranking with recency fallback; 50 active entries are considered, at most 20 and 6,000 characters reach prompts.
- Added semantic JSON export API/UI and tablet Studio toggle.
- Added transactional, append-only episode review history plus local operator review CLI.
- Added scheduler-compatible purge CLI covering working, episodic, and semantic memory.
- Full baseline passed 190 tests; memory evaluation passed all nine checks with 0.454 ms p95 local recall.
- Remaining decisions: reviewer/admin web authorization, multi-host database, external scheduler wiring, label-blind model usefulness evaluation, and any later fine-tuning.

## Active work — Step 15 trusted review and ranking evidence

### Requirements

- Derive reviewer authority from server configuration, never JSON bodies or browser state.
- Require reviewer to remain a member of reviewed workspace.
- Prevent owner forgery and concurrent double-review.
- Keep review audit and status transition atomic.
- Evaluate semantic ranking with separate inference tasks and labels.

### Result — 28 October 2026

- Added `DOCQA_WORKSPACE_REVIEWERS`, server-derived `can_review`, pending review API, and conditional Studio queue.
- Added atomic unverified-only approve/reject path; server resolves episode owner and records authenticated reviewer.
- Invalid/non-member reviewer configuration fails with 503; ordinary members receive 403.
- Added development-blind memory-retrieval tasks/labels and label-leakage rejection; Recall@1 reached 1.0.
- Expanded semantic-memory rejection for secret assignments, credential-bearing URLs, and JWT-like tokens.
- Full baseline passed 194 tests and seven stages. Fine-tuning remains paused.

## Active work — Step 16 portable memory migration

### Requirements

- Define backend-neutral memory rows before implementing another database.
- Reject inconsistent/running state and validate every relationship.
- Make snapshots tamper-evident, private, and non-overwriting.
- Rehearse only on a copy and prove source immutability.

### Result — 28 October 2026

- Added six-table versioned JSON contract with canonical SHA-256.
- Added completed-turn, primary-key, conversation/turn/episode/review, status, and JSON validation.
- Added private `0600` snapshot writer and copy-only SQLite backup rehearsal.
- Full baseline passed 196 tests and seven stages.
- PostgreSQL adapter/runtime switch remains blocked until a staging database and rollback policy exist.
