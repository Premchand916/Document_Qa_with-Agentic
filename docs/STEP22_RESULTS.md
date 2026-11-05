# Step 22 Results — Governed Connector to Chat

**Status:** managed-folder connector to chat completed locally; external provider connectors remain later work.  
**Date:** 5 November 2026

## Objective

Replace direct browser upload as the only knowledge transport with the first governed, read-only connector path. The slice had to prove ownership, bounded extraction, immutable history, retries, revocation, deletion, and safe API failures before connector content can influence answers.

## Implemented

- Added typed managed-folder connector, source, file, scan, and sync contracts.
- Added a server-managed read-only folder adapter. Clients provide a validated relative source key, never a filesystem path.
- Bounded each source to 20 supported files, 2 MiB per file, 10 MiB total, and five nested path levels. Durable state is capped at 10 sources per owner and 10 immutable snapshots per source; the newest 200 completed sync records are retained. Empty sources are valid deletion states. Symlinks, hardlinks, path traversal, unsupported files, missing roots, and changed-size reads fail closed.
- Added owner-scoped SQLite grants, sources, sync runs, cursors, immutable snapshots, and snapshot file records.
- Added idempotent source creation and sync. An unchanged cursor reuses the current snapshot; a source returning to an older state reuses the matching immutable snapshot.
- Added local-mode authorization and explicit principal ACLs for shared access. A shared principal cannot register or sync an unlisted server-managed source.
- Added freshness states: `never_synced`, `fresh`, `stale`, and `revoked`.
- Added immediate grant revocation. Revoked snapshots cannot be listed or loaded, including if revocation races with sync completion.
- Added permanent source deletion that removes copied files, snapshots, sync history, the source, and its dedicated grant.
- Added safe APIs for source registration/listing, synchronous sync, snapshot inspection, grant revocation, and source deletion.
- Added provenance on copied files: source ID, snapshot ID, relative path, SHA-256, media type, byte count, cursor, and snapshot timestamp.
- Added explicit `publish` after sync. Each immutable snapshot is parsed in isolation, chunked, embedded, and stored in a separate principal/source index. Publication becomes visible only after indexing succeeds and the source state is rechecked.
- Federated local-upload and authorized connector indexes at read time. A new local upload or connector publication cannot replace the other source. A changed sync requires another explicit publish; an empty published snapshot removes its content from chat.
- Carried connector source and snapshot IDs through chunks, table assets, chat results, and citations. The Knowledge Sources UI now supports connect, sync and publish, revoke, and delete.
- Rechecked grant, ACL, publication, freshness, and revocation epoch before saving a reply. Revocation/deletion removes derived indexes and redacts stored connector-derived answers and replay events. ACL removal also redacts on conversation read or next chat.

## API and configuration

Set an absolute server-owned folder:

```dotenv
DOCQA_CONNECTOR_ROOT=/absolute/path/to/managed-knowledge
DOCQA_CONNECTOR_FRESHNESS_SECONDS=86400
```

Local loopback mode may use any valid source key below that root. Shared auth-proxy deployments must explicitly map source keys to principals:

```dotenv
DOCQA_CONNECTOR_SOURCE_ACL={"operations/sop":["alice@example.com"],"sales/sow":["bob@example.com"]}
```

Available endpoints:

- `POST /api/connectors/sources`
- `GET /api/connectors/sources`
- `POST /api/connectors/sources/{source_id}/sync` with `Idempotency-Key`
- `POST /api/connectors/sources/{source_id}/publish`
- `GET /api/connectors/sources/{source_id}/snapshots`
- `DELETE /api/connectors/grants/{grant_id}`
- `DELETE /api/connectors/sources/{source_id}`

No connector credential is accepted or persisted in this adapter. Future OAuth connectors must store only a vault reference, never a token, in application storage.

## Verification

Focused connector and regression checks cover:

- registration → sync → inspect → revoke → hard delete through the API;
- idempotent retries, unchanged data, changed data, deletion, and return to an earlier cursor;
- restart persistence, owner isolation, shared ACL denial, freshness, and revocation;
- traversal, symlink, hardlink policy, unsupported-file, missing-source, file-size, durable-capacity, and safe-error behavior.
- local upload plus connector retrieval, cited answer provenance, reupload isolation, changed/empty snapshot publication, owner isolation, ACL removal, and historical answer/event redaction.

Full baseline: 231 tests and all seven stages pass. Focused connector suite: 12 tests. The inline UI scripts pass a JavaScript syntax check. Browser interaction could not run because the local browser control host was unavailable and local port binding was denied in this environment.

## Remaining boundaries

This local slice has no external SaaS or OAuth provider, vault, background scheduler, incremental provider API, or live read. Publication is synchronous and bounded by the adapter's file limits, but it is not a queued background job. SQLite and the in-process reply lock are single-host assumptions; multi-host deployment requires coordinated storage and revocation. Connector search scores assume compatible embeddings across the local and connector indexes. Real-device browser checks remain outstanding. Fine-tuning remains paused.
