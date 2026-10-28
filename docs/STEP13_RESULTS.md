# Step 13 results — four-layer memory architecture

Verified 28 October 2026. Memory foundation now has explicit ownership and safety boundaries. Ollama fine-tuning remains paused.

## Architecture decision

- **Working memory is required.** It carries current conversation context, follow-up references, and a bounded recap. `ConversationRepository` remains sole durable transcript owner: eight recent turns, 16,000 prompt characters, 6,000 recap characters, scoped leases, retention, deletion, and pagination.
- **Semantic memory is useful but must be explicit.** It stores stable preferences, profile facts, constraints, and terminology across conversations. It is never extracted automatically from chats and never counted as document evidence.
- **Episodic memory is useful only for selected outcomes.** Existing episodes remain tied to completed turns, evidence/corpus versions, explicit corrections, verification state, owner, task key, and expiry. Ordinary conversation turns do not become episodes.
- **Procedural memory is required for agent behavior, but it is not user memory.** Procedures remain version-controlled graph routes, allowlisted tools, planners, evidence rules, and operator-selected prompt skills. The model cannot rewrite or approve its own procedures.

## Perspective analysis

- **Conversation quality:** working memory gives continuity; semantic memory prevents repeated preference setup; episodes reuse verified corrections; procedures keep tool use repeatable.
- **RAG correctness:** documents and live web results remain the only answer evidence. Semantic and episodic memory may guide style or process but cannot satisfy citations.
- **Security:** identity comes from the server boundary. Semantic memory rejects credential-like values and instruction-override text. Procedural assets are repository-owned or explicitly configured by an operator; arbitrary Downloads-folder discovery was removed.
- **Privacy:** semantic memory is opt-in, inspectable, replaceable, deletable, owner/workspace scoped, limited to 50 active entries, bounded to 6,000 recalled characters, and expires after 365 days by default.
- **Operations:** all current memories share local SQLite durability but separate tables and lifecycle rules. This supports shared local workers, not multi-host deployment.
- **Model training:** memory content is runtime context, not training data. Fine-tuning later requires separate consent, redaction, provenance, deduplication, and held-out evaluation.

## Implemented

- Added `SemanticMemoryRepository` with `preference`, `profile`, `constraint`, and `terminology` categories.
- Added owner-scoped upsert/list/delete APIs and startup expiry cleanup.
- Added prompt integration with an explicit untrusted/non-evidentiary boundary.
- Added browser controls to create, inspect, and delete semantic memory.
- Added secret-pattern and persistent prompt-override rejection.
- Semantic memory survives conversation deletion by design; episodic memory continues to cascade with its source conversation.
- Procedural prompt skills now have SHA-256 identity and repository/operator source labels. Unapproved home Downloads discovery is removed.

## Verification

- Repository tests cover upsert identity, replacement, owner/workspace isolation, expiry, deletion, conversation independence, secret rejection, instruction-override rejection, and retention bounds.
- API tests prove clients cannot forge principals, semantic memory reaches prompt guidance, and it never becomes documents or citations.
- UI tests prove create/list/delete controls and warnings exist.
- Procedural tests prove default discovery is repository-only and explicitly configured procedures have stable SHA-256 identity.
- Frozen deterministic evaluation `docqa-memory-foundation-2026-10-v1` passes all eight gates: explicit recall, latest-value conflict resolution, owner/workspace isolation, expiry, deletion, injection rejection, non-evidentiary prompt separation, and recall latency.
- Full offline baseline passes 187 tests. Diagnostics report zero reproduced defects, 25 not reproduced, one passing rejection control, zero failed controls, and zero probe errors. Ingestion, index, memory, and environment stages pass.
- Memory recall measured 0.441 ms p95 across 100 local SQLite iterations against a 50 ms development ceiling. This is a local micro-measurement, not a production SLO.

## Remaining limits

- Semantic recall currently loads bounded recent entries rather than ranking by task relevance. Add relevance only after a labeled memory-evaluation set exists.
- No trusted reviewer/admin role exists for approving dynamic procedures. Therefore no writable procedural-memory API exists.
- Episode verification still uses an internal repository boundary; reviewer UI and audit trail remain future work.
- Startup-triggered expiry is adequate locally; production needs scheduled cleanup and deletion/export governance.
- Multi-host operation needs shared transactional storage instead of local SQLite.

The deterministic memory foundation is complete. Fine-tuning stays paused for this workstream; the next memory evidence should be a separate real-model, label-blind evaluation of answer usefulness and evidence conflicts, not training on saved memory.
