# Step 20 Results — Capability and Event Kernel

**Status:** implemented and verified locally  
**Date:** 3 November 2026

## Built

- Versioned capability contracts for tools, skills, plugins, and connectors with bounded public JSON schemas and explicit risk/scopes.
- Deterministic conflict-safe registry and fail-closed policy filtering. Denied definitions are absent from model-visible contracts.
- `assistant.chat` registration for the proven existing workflow and a small terminal turn-state contract.
- Safe append-only turn events with conversation-wide ordering, per-turn deduplication, size/depth/type limits, and secret-key rejection.
- Atomic `turn.started`, `message.completed`, `turn.completed`, and `turn.failed` persistence with the authoritative conversation transaction.
- Owner-scoped cursor replay at `GET /api/conversations/{conversation_id}/events`.
- Idempotent durable cancellation requests plus replayable `turn.cancel_requested` events.
- Expired-lease failure events, restart-safe replay, and cascade deletion with conversations.

## Validation

Focused verification passes 26 tests across capability policy, events, conversations, API replay, restarts, isolation, expiration, cancellation, and deletion. Compile and whitespace checks pass. The full baseline passes 212 tests and all seven stages: tests, diagnostics, ingestion, index benchmark, memory, memory retrieval, and environment.

## Security and data decisions

- Capability definitions grant no permissions and contain no executors.
- Policy is applied before building future model-visible capability contracts.
- Event responses omit workspace/principal identifiers and expose only authorized streams.
- Events store safe observable state, not hidden reasoning or credentials.
- Lifecycle changes and their events commit atomically, preventing a completed answer without its terminal event.

## Deliberate limits

- Replay uses bounded polling; SSE, reconnect rendering, and hard Stop are Step 23.
- Cancellation currently records intent but does not interrupt native/model work.
- The current chat remains on its proven path; the bounded dynamic tool loop arrives in Step 24.
- Skills, connectors, plugin/MCP, coding tools, and subagents remain disabled until their ordered steps.

## Next gate

Step 21: load versioned repository-local skills safely, starting with prompt profiles. Validate metadata, paths, symlinks, digests, enablement, dependency declarations, progressive content loading, and activation audit events. Skill text must never grant capabilities.
