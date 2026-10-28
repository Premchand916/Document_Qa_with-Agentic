# Step 14 results — memory limitation hardening

Verified 28 October 2026. This step closes limitations that fit the current local architecture. Ollama fine-tuning remains paused.

## Mental model

```text
Current turn + bounded transcript  -> working memory
Explicit stable user facts         -> semantic memory
Reviewed corrections/outcomes      -> episodic memory
Versioned graph/tool/prompt rules   -> procedural memory

Only document/web/tool output       -> answer evidence and citations
Memory                              -> context and behavior guidance only
```

These stores are separate because they have different owners, trust levels, recall rules, retention periods, and deletion behavior. Combining them in one generic vector index would make authorization, stale-data handling, and evidence boundaries harder to prove.

## 1. Semantic recall: recency-only to query-aware

### Previous limitation

Recall selected the 20 newest active memories. An older warranty term could disappear from the prompt behind newer formatting preferences.

### Implementation

- Scan at most 50 active, already owner/workspace-authorized records.
- Normalize query, key, and value into deterministic terms; remove common query stop words.
- Score key overlap at four points and value overlap at one point.
- Sort by relevance, then update time, then stable memory ID.
- Preserve the existing 20-record and 6,000-character prompt limits.
- Use recency order when no query terms exist.

No embedding model or network call was added. This keeps latency, reproducibility, and local operation predictable. The frozen memory gate now includes an older-relevant-versus-newer-irrelevant case.

## 2. Semantic export and responsive control

### Previous limitation

Users could inspect and delete entries individually but could not export them. The Studio panel was hidden at tablet widths.

### Implementation

- Added owner-scoped `GET /api/memories/semantic/export`.
- Export contains schema version, kind, export time, workspace, and all active semantic records.
- Added JSON download control in Studio.
- Added a keyboard-accessible Memory toggle for the 761–1180 px layout; mobile retains the existing stacked Studio panel.

The export intentionally excludes secrets, conversations, and procedural configuration. It exports only the authenticated user's semantic memory in the active workspace.

## 3. Episodic review audit

### Previous limitation

Episode verification was an internal status update with no durable reviewer history.

### Implementation

- Added append-only `episode_reviews` rows containing reviewer ID, previous status, decision, note, and timestamp.
- Review and episode updates run in one SQLite transaction.
- Reviews remain owner/workspace scoped and cascade when their source episode is deleted.
- Public API still cannot self-verify feedback.
- Added local operator command `python3 -m tools.review_episode`.

This provides traceability without exposing a fake administrator API before trusted roles exist.

## 4. Scheduled cleanup target

### Previous limitation

Physical deletion happened mainly at application startup. Expired entries were hidden from recall, but inactive deployments could retain rows longer than intended.

### Implementation

- Added `python3 -m tools.purge_memories` for cron, systemd, Kubernetes CronJob, or another deployment scheduler.
- One run purges expired working conversations, episodic memory, and semantic memory.
- Tool reports per-store deletion counts as JSON.
- Existing transaction, running-turn, foreign-key, and retention safeguards remain active.

The repository does not install an OS scheduler because schedule ownership depends on deployment environment.

## Validation

- Full offline baseline: 190 tests, all six stages passed.
- Diagnostics: zero reproduced defects, 25 not reproduced, one rejection control passed, zero failed controls, zero probe errors.
- Frozen memory evaluation: all nine checks passed, including relevance ranking.
- Local SQLite semantic recall: 0.454 ms p95 over 100 iterations against a 50 ms development ceiling.
- Focused API/repository tests cover query ranking, export ownership, audit history, cleanup, and UI controls.
- Desktop UI visually loaded with semantic create/export controls; responsive selectors and toggle behavior are regression-checked in the static UI test.
- Syntax compilation and whitespace validation passed.

## Remaining boundaries

1. **Trusted web reviewer role:** not built. Current authentication defines membership, not reviewer/admin authorization. Add signed roles and server-side authorization before review APIs or UI.
2. **Multi-host persistence:** not built. Local SQLite is correct for current deployment. Move repository interfaces to PostgreSQL only when multi-host topology and migration/rollback requirements are known.
3. **External schedule wiring:** purge target exists; production deployment must invoke it and monitor failures.
4. **Semantic ranking quality:** lexical ranking is deterministic but does not understand synonyms. Compare it with embedding or learned ranking on a label-blind set before changing it.
5. **Secret detection:** pattern rejection is defense-in-depth, not a complete data-loss-prevention system. UI warning, explicit opt-in, access control, and deletion remain required.
6. **Model behavior:** deterministic tests prove data boundaries, not answer usefulness. Run a label-blind Ollama memory evaluation before fine-tuning or using memory traces as training data.

## Operator examples

```bash
python3 -m tools.evaluate_memory

python3 -m tools.purge_memories \
  --data-dir vector_db \
  --conversation-retention-days 90 \
  --episode-retention-days 180 \
  --semantic-retention-days 365

python3 -m tools.review_episode \
  --data-dir vector_db \
  --workspace local \
  --principal local-developer \
  --episode-id EPISODE_ID \
  --decision verified \
  --reviewer reviewer@example.com \
  --note "Evidence and calculation checked."
```
