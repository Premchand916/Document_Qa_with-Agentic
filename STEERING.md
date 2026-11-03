# Build Steering and Continuity

**Updated:** 3 November 2026  
**Current objective:** build the capability platform in dependency order without weakening the proven document, memory, ownership, or evidence boundaries.

## Read this first

This is the compact restart checkpoint. At the start of every build task, read this file and the repository `AGENTS.md`. After each material task, update the completed work, verification, limitations, and exact next action here. Detailed evidence stays in `docs/STEP*_RESULTS.md`; do not inflate this file with logs.

## Fixed constraints

- Use a modular monolith and existing repository interfaces; no big-bang rewrite.
- Keep workspace/principal ownership server-derived and fail closed.
- Cap payloads, time, concurrency, context, and durable retention.
- Never expose credentials, hidden reasoning, or unfiltered capabilities to a model.
- Keep retrieved and connector content untrusted and cite factual evidence.
- Voice and Wispr integration are deferred. Wispr text currently enters as ordinary composer text.
- Ollama fine-tuning stays paused until capability, evidence, and evaluation failures are separated from model-behavior failures.
- Do not commit, push, deploy, install plugins, or contact third parties without explicit user instruction.

## Verified completed work

- Steps 0–19: bounded ingestion, versioned corpora, routing/calculation, durable conversations, hybrid retrieval, evidence contracts, memory layers, evaluation tooling, and chat-first product shell.
- Architecture review: Steps 20–31 and security boundaries documented in `docs/CHATGPT_CAPABILITY_REVERSE_ENGINEERING.md`.
- Step 20 code: typed capability definitions; deterministic registry; fail-closed policy filtering; registered legacy chat capability; canonical current turn states; append-only SQLite events; atomic lifecycle events; owner-scoped cursor replay API; durable cancellation requests; expiry/restart recovery; secret-key rejection.
- Step 20 verification: 26 focused tests pass; the full baseline passes 212 tests and all seven stages.

## Current files

- Contracts: `domain/capabilities.py`, `domain/events.py`, `runtime/turns.py`
- Registry/policy: `capabilities/registry.py`, `capabilities/builtin.py`, `policy/capabilities.py`
- Events: `runtime/events/repository.py`, `storage/conversations.py`
- API: `GET /api/conversations/{conversation_id}/events` in `app/server.py`
- Tests: `tests/test_capability_registry.py`, `tests/test_turn_events.py`
- Decision/result: `docs/adr/0001-capability-kernel.md`, `docs/STEP20_RESULTS.md`

## Current limitations

- Replay is bounded HTTP polling, not SSE streaming.
- A cancellation request is durable and visible, but hard Stop and a `cancelled` terminal state belong to Step 23.
- Capability policy exists, but a dynamic model/tool loop does not; Step 24 will consume filtered contracts.
- No skill packages, external connectors, plugin/MCP calls, shell execution, coding sandbox, or subagents exist yet.
- SQLite/local mode is not multi-host production infrastructure.

## Exact next action

Implement Step 21 only: versioned repository-local skills with strict metadata/path/symlink validation, digest locks, progressive loading, workspace enablement, and audit events. Convert the current prompt profile into a built-in skill without changing answer behavior.

## Verification commands

```bash
python3 -m unittest tests.test_capability_registry tests.test_turn_events tests.test_conversations -v
python3 -m compileall -q capabilities domain policy runtime storage app tests
python3 -m tools.run_baseline
git diff --check
git status --short
```

## Update protocol

For each task, record: objective; constraints changed; files/behavior completed; checks and counts; limitations/blockers; exact next action. If context is compacted, continue from this file and current `git diff`; do not restart completed analysis.
