# Step 6 results — durable conversations

Verified 28 October 2026. Local acceptance gate passes for one or more API workers sharing the same SQLite file. Distributed or cross-region deployment is outside this gate.

## Implemented

- SQLite remains the authoritative transcript under server-owned workspace and principal identity. Client-supplied history is validated but never trusted as memory. Completed idempotent requests replay their stored response.
- Running turns use opaque 60-second leases. API heartbeat renews every 20 seconds. Only the live lease token may renew, complete, or fail a turn. Expired leases can be reclaimed; stale workers cannot overwrite or fail the replacement. Startup recovers expired leases only, so a second worker no longer invalidates active work.
- Existing databases migrate additively with `lease_token` and `lease_expires_at`; saved conversations remain readable. SQLite WAL and a bounded busy timeout support shared local workers.
- Conversation deletion is workspace/principal scoped and transactionally removes summaries, turns, and the conversation. UI requires explicit confirmation.
- Default retention is 90 days and runs at startup. `DOCQA_CONVERSATION_RETENTION_DAYS` accepts 1–3650 days. Active running turns are excluded. `DOCQA_TURN_LEASE_SECONDS` accepts 10–3600 seconds.
- Prompt memory keeps the latest eight completed turns and 16,000 characters. Older completed turns contribute through a deterministic extractive recap capped at 6,000 characters; original turns remain until deletion or retention cleanup.
- Conversation detail returns at most 50 turns and supports a `before_sequence` cursor. UI loads older pages on demand.
- Context-free references such as “What is its deadline?” return `needs_clarification` without retrieval or model execution. Valid follow-ups still expand retrieval from persisted user context.
- Desktop preview verified conversation creation and permanent deletion. Tablet/mobile CSS now collapses the three-column layout into bounded two-column and stacked layouts.

## Verification

- `python3 -m unittest discover -s tests -t . -q`: **144 tests pass**.
- Conversation tests cover additive migration, restart persistence, principal/workspace isolation, idempotent replay, live-lease conflict, renewal, expiry/reclaim, stale-token rejection, startup safety, recap construction, cursor pagination, deletion, retention, API deletion, and context-free clarification.
- `python3 -m tools.audit_failure_modes`: zero reproduced defects, 25 not reproduced, one passing rejection control, zero probe errors.
- Live synthetic browser preview: create conversation → Delete control appears → confirmation appears → deletion returns to empty state.
- `python3 -m compileall`, `git diff --check`, ingestion/OCR measurement, embedding measurement, and environment inspection pass through the full baseline runner.

## Remaining limits

- Shared workers must use the same local SQLite database. Network filesystems, multiple hosts, and cross-region leases require PostgreSQL or another transactional shared database plus database-time lease semantics.
- Extractive recap is deterministic and auditable but not a semantic summary. Evaluate a local summarizer before replacing it.
- LangGraph state still contains live FAISS/Pandas objects and remains request-scoped. SQLite owns transcript durability; adding a graph checkpointer now would create competing state ownership.
- Worker threads cannot forcibly cancel native/model work. Durable jobs, hard deadlines, streaming status, stage metrics, and per-workspace fairness remain Step 10.
- Retention default is implemented but still needs deployment-specific legal/product approval. Purging runs at startup, not continuously.
- Responsive rules have automated contract coverage and desktop browser verification; physical-device/mobile-browser validation remains release work.

Next: Step 9 opt-in episodic feedback with verified/unverified separation, expiry, authorization, and deletion propagation. Then Step 10 durable jobs and operational load gates.
