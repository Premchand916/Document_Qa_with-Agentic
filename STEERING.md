# Build Steering and Continuity

**Updated:** 5 November 2026
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
- Step 20 verification: 26 focused tests pass.
- Step 21: strict repository-local skill packages, digest locks, progressive loading, workspace enablement/audit, policy-bounded dependencies, explicit activation, prompt-library skill, read-only API, and activation events.
- Reusable `build-application-safely` skill v1.1: risk mapping, vertical slices, failure recovery, layered verification, truthful evidence boundaries, durable handoff, and canonical `resources/development-checklist.md`.
- Step 22: managed-folder connector to chat with explicit publish, isolated principal/source FAISS indexes, read-time composition with local uploads, snapshot provenance and citations, ACL/freshness/epoch checks, revocation cleanup, and stored-answer/event redaction.
- Step 22 focused connector suite: 12 tests pass, including reupload composition, empty/failed publication, owner isolation, and ACL revocation.
- Step 23: asynchronous chat turns, owner-scoped SSE replay and Last-Event-ID reconnect, bounded progress streaming, final validated answer retrieval, and atomic terminal Stop with an in-place turn-table migration.
- Step 23 focused event suite: 11 tests pass, including legacy migration, asynchronous completion, and a Stop race against a blocked graph result.
- Step 24 first local slice: opt-in general-chat tool loop with policy-filtered math/UTC tools, strict model actions, call/step/estimated-token/time limits, safe tool events, and explicit owner approval for semantic-preference writes.
- Step 24 focused tool suite: 11 tests pass; isolated Chrome showed pending approval, approved state, and immediate Memory count refresh.
- UI follow-up: higher-contrast chat shell; source-aware starter prompts; managed-source-first Knowledge Sources dialog with visible error/status states; connector provenance labels; keyboard composer newline; modal focus handling; mobile panel backdrop and responsive source form. Local-file import remains under a secondary disclosure.
- Current full baseline: 245 tests and all seven stages pass.
- Local API smoke: `/api/skills` and `/api/prompt-library` return 200 with locked skill metadata. Chrome UI checked at desktop and 393px mobile width; starter focus, multiline composer, modal open/Escape return, mobile navigation, and inline connector validation error observed. Step 23 Chrome check used isolated temporary data and a slow synthetic model: progress, Stop button, and durable “Stopped.” message were observed.

## Current files

- Contracts: `domain/capabilities.py`, `domain/events.py`, `runtime/turns.py`
- Registry/policy: `capabilities/registry.py`, `capabilities/builtin.py`, `policy/capabilities.py`
- Events: `runtime/events/repository.py`, `storage/conversations.py`
- Tool loop/approval: `runtime/tool_loop.py`, `capabilities/read_tools.py`, `storage/tool_approvals.py`, `memory/semantic_memory.py`
- API: asynchronous turns, SSE replay, Stop, and bounded HTTP replay in `app/server.py`
- Tests: `tests/test_capability_registry.py`, `tests/test_turn_events.py`, `tests/test_tool_loop.py`
- Skills: `domain/skills.py`, `capabilities/skills/`, `.agents/skills/`; canonical development checklist at `.agents/skills/build-application-safely/resources/development-checklist.md`
- Skill test: `tests/test_skill_runtime.py`
- Connectors: `domain/connectors.py`, `connectors/managed_folder.py`, `connectors/service.py`, `storage/connectors.py`, connector routes in `app/server.py`
- Connector test: `tests/test_connectors.py`
- Decisions/results: `docs/adr/0001-capability-kernel.md`, `docs/adr/0002-versioned-skill-runtime.md`, `docs/STEP20_RESULTS.md`, `docs/STEP21_RESULTS.md`, `docs/STEP22_RESULTS.md`, `docs/STEP23_RESULTS.md`, `docs/STEP24_RESULTS.md`

## Current limitations

- Streamed progress is phase/tool status and a final validated answer; raw model token deltas need a provider streaming adapter and a safe verification protocol.
- Stop prevents late publication but cannot forcibly interrupt an already running synchronous model thread; the deadline and chat-slot cap bound it.
- SSE and background tasks use a single-host in-process worker and SQLite; multi-host delivery needs a shared queue and event store.
- Step 24's model/tool loop runs only in opt-in general chat. Grounded document/web answers still use the evidence-validated graph; they must not be bypassed by a model-selected final action.
- Token limits are exact only when a provider reports usage; otherwise they are estimated from text. The per-loop time check cannot abort a currently blocking provider call.
- Preference writes are the only approved write executor. Additional writes and consequential actions need dedicated validation and approval contracts.
- Managed-folder publication is synchronous; there is no remote SaaS connector, OAuth/vault integration, background connector scheduler, plugin/MCP call, shell execution, coding sandbox, or subagent yet.
- Skills are repository-local UTF-8 guidance only. Remote installation, executable scripts, and automatic production routing are intentionally disabled.
- SQLite/local mode is not multi-host production infrastructure.
- UI validation used local managed-folder configuration and a deliberately invalid source key; live model-response quality and real remote connector behavior were not part of this visual pass.

## Exact next action

Complete Step 24's grounded document/web tool integration through existing citations, source revocation, and evidence checks. Evaluate tool selection on local Ollama before adding plugin/MCP capabilities. Keep Ollama fine-tuning paused.

## Verification commands

```bash
python3 -m unittest tests.test_connectors tests.test_skill_runtime tests.test_capability_registry tests.test_turn_events tests.test_conversations -v
python3 -m compileall -q capabilities domain policy runtime storage app tests
python3 -m tools.run_baseline
git diff --check
git status --short
```

## Update protocol

For each task, record: objective; constraints changed; files/behavior completed; checks and counts; limitations/blockers; exact next action. If context is compacted, continue from this file and current `git diff`; do not restart completed analysis.
