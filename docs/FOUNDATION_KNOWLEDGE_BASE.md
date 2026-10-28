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
