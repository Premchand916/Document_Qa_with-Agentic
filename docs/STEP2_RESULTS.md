# Step 2 — Workspace Storage and Atomic Publication

Implemented and verified 26 October 2026. **71 offline tests pass.** Five more diagnostic failures no longer reproduce: S01, S02, P01, P04, and A04. There are **17 remaining reproduced limitations**, eight repaired cases, one passing rejection control, and zero probe errors. Original baseline and Step 1 evidence remain preserved.

## How the implementation works

1. **Authorize the workspace.** `app/security.py` validates the trusted proxy identity and server-configured membership before parsing the request. `X-Workspace-ID` selects a permitted workspace; it cannot grant access. Local mode still represents one developer and rejects identity/workspace overrides.
2. **Capture the starting state.** `app/server.py` reads the workspace provider and active corpus version from SQLite. Provider choice travels explicitly through question generation and each model-calling graph node. Ollama fallback no longer changes process environment settings.
3. **Build a replacement.** Parsing, chunking, and embedding produce a new in-memory index. Any embedding failure rejects the whole upload, including mixed text/CSV uploads; the previous corpus stays active.
4. **Write a private version.** `storage/workspaces.py` writes native FAISS vectors, JSON documents, schema-bearing Pandas tables, questions, file types, and source names/SHA-256 hashes. Files are flushed; a manifest records checksums, vector dimensions/counts, and table counts. The repository reconstructs the saved artifacts before publication.
5. **Publish transactionally.** A short SQLite `BEGIN IMMEDIATE` transaction compares the current version with the upload's starting version, then commits the new manifest hash and active pointer together. Two uploads based on the same version cannot both publish; the loser receives **409**. A failed artifact write or pre-commit exception preserves the old pointer.
6. **Pin each reader.** Chat loads one complete version and provider before invoking the graph, returning the version ID with its answer. Replacement cannot mix its vectors and tables. Old versions remain on disk, including when publication succeeds but its caller loses the response.

Default storage layout:

```text
vector_db/
  workspaces.sqlite3
  corpora/<SHA-256 of workspace ID>/<version UUID>/
    index.faiss
    documents.json
    tables.json
    corpus.json
    manifest.json
```

`DOCQA_DATA_DIR` optionally changes the FastAPI storage root. Keep it private to the application account and outside version control. Back up SQLite and its referenced artifact directories together while writes are stopped. No cleanup, rollback endpoint, schema-upgrade migration, or retention policy is implemented yet.

## Configuration and migration

Keep the authentication-proxy settings from [Step 1](STEP1_RESULTS.md). For separate workspaces, add this server-side membership mapping using stable authenticated subject IDs:

```dotenv
DOCQA_WORKSPACE_MEMBERSHIPS='{"alpha":["alice-id"],"beta":["bob-id"],"shared":["alice-id","bob-id"]}'
DOCQA_WORKSPACE_ID=alpha
```

When the mapping is absent, `DOCQA_ALLOWED_SUBJECTS` retains the single-workspace configuration. Members of the same workspace intentionally share its corpus and provider and can replace it; there are no read-only roles. The backend validates every selection. The proxy must preserve the selection header while overwriting identity/secret headers as described in Step 1. Actual proxy/identity-provider deployment remains unverified.

**Legacy `index.pkl` files are never deserialized or migrated automatically.** Existing files stay untouched. The state API/UI reports the legacy index and requests original-file re-upload. The legacy Streamlit storage wrapper uses a separate `legacy-default` namespace and remains outside the FastAPI access boundary.

## UI and verification

The workspace selector lists authorized workspaces. Switching clears browser chat, source cards, notes, and draft input before loading the destination. Switching, uploading, sending, and provider changes are guarded while another operation is in progress. Reload restores sources, questions, and provider; it selects the server's default permitted workspace. Provider buttons expose their pressed state to accessibility tools. Duplicate Alpine initialization was removed.

```bash
python3 -m tools.run_baseline --output-dir .baseline-results/step2
python3 -m unittest tests.test_workspace_storage tests.test_provider_scope -v
python3 -m tools.preview_step1 --port 8765 --workspaces
```

Evidence: [test summary](step2/summary.json), [diagnostics](step2/diagnostics.json), [browser checks](step2/browser_checks.json). The 21 added tests cover authorization, restart reconstruction, nullable integer tables, corrupted artifacts, conflicting uploads, failures before/after commit, pinned graph context, provider propagation, and environment isolation. Concurrent publication uses two threads with independent repositories against the same SQLite database. Failure tests inject exceptions; they do not simulate power loss or prove multi-process production support.

Chrome verified alpha document QA, note clearing when switching to beta, beta CSV upload and total **30**, and persisted beta/Gemini versus alpha/Ollama after reload. The preview uses temporary artifacts and synthetic models/identity; it does not authenticate real users. At **393 × 852**, the existing three-column layout still hides the chat panel; this remains a recorded UI defect.

## Remaining limits and next step

- Workspace requests reload artifacts and build BM25 locally instead of retaining corpus objects in unbounded global registries. This favors isolation; latency/RAM benchmarks and bounded version-keyed caches remain Steps 7/10.
- JSON tables preserve tested nullable integer schemas; arbitrary Pandas extension types and exact high-precision numeric/timestamp fidelity are not guaranteed. Artifact decoding or validation errors prevent publication. Existing parser dtype loss and incorrect table routing/filtering remain unresolved.
- Manifests check stored bytes and shape, not encoder identity/normalization compatibility (Step 4). Checksums assume a trusted SQLite database and filesystem owner; they are not signatures against an attacker controlling both.
- Original upload bytes are not retained. Old/unpublished versions accumulate. Parser resource bounds, disk quotas, durable conversations, episodic memory, and live-model quality remain outstanding.

**Next: Step 3 — bounded ingestion and complete provenance.** Parse each source once, preserve numeric types, split every format within token limits, retain page/sheet/row/chunk references, and reject extraction/resource failures explicitly.
