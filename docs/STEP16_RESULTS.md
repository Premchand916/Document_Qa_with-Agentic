# Step 16 results — portable memory migration contract

Verified 28 October 2026. Runtime storage remains SQLite; no unverified PostgreSQL switch occurred. Ollama fine-tuning remains paused.

## What was built

- Versioned backend-neutral JSON contract covering conversations, completed turns, summaries, semantic memories, episodes, and episode-review audit rows.
- One consistent SQLite read transaction for snapshot creation.
- Running turns block export. Completed turns retain responses; failed turns retain failure state without requiring a response.
- Exact table/column, primary-key, conversation, turn, episode, review, status, and JSON-field validation.
- Canonical SHA-256 over contract payload; generated timestamp does not change data identity.
- Private snapshot creation with mode `0600` and no-overwrite behavior.
- Copy-only rehearsal using SQLite backup, source/copy checksum comparison, and second source snapshot proving no mutation.

## Why this comes before PostgreSQL

PostgreSQL code without a deployed database, connection policy, migration window, or rollback target would be unverified infrastructure. This contract defines what must survive any later importer:

- ownership and workspace isolation;
- completed transcript values;
- semantic-memory identity and expiry;
- episode-to-turn provenance;
- reviewer audit history;
- stable counts and checksum evidence.

Later PostgreSQL work must import this contract into staging, validate relationships/counts/checksum, exercise concurrent writers, and rehearse rollback before runtime selection changes.

## Operator command

```bash
python3 -m tools.rehearse_memory_migration \
  --data-dir vector_db \
  --snapshot-output /secure/path/memory-snapshot.json \
  --report-output .baseline-results/memory-migration.json
```

Snapshot contains conversation and memory content. Keep it encrypted/private, never commit it, and delete it according to retention policy.

## Verification

- Full baseline: 196 tests; all seven stages passed.
- Diagnostics: zero reproduced defects, 25 not reproduced, one passing rejection control, zero failed controls, zero probe errors.
- Tests prove relational validation, audit preservation, copy equality, source immutability, running-turn rejection, checksum tamper rejection, `0600` output, and overwrite refusal.
- Compilation and whitespace checks pass.

## Remaining boundaries

- PostgreSQL adapter/importer and live database tests are not built.
- Online dual-write/cutover is not designed; do not run two storage authorities.
- Snapshot encryption/key management belongs to deployment platform.
- Active-writer quiescing and rollback approval need production operations policy.
- Identity-provider deployment and label-blind Ollama usefulness evaluation remain open.
