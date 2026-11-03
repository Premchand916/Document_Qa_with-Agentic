# ADR 0001: One capability and event kernel

**Status:** accepted locally  
**Date:** 3 November 2026

## Context

The application already has reliable chat, document retrieval, calculations, web search, memory, and durable conversations. Skills, connectors, plugins, coding tools, and subagents would create incompatible authorization and audit paths if each added its own contracts.

## Decision

- Represent tools, skills, plugins, and connectors with one versioned `CapabilityDefinition`.
- Treat registration as description, never authority. Server policy filters disabled, unavailable, unscoped, and disallowed-risk capabilities before model exposure.
- Register the existing chat workflow as `assistant.chat`; do not rewrite its execution path in this step.
- Persist safe, user-visible events in the same SQLite transaction as turn lifecycle changes.
- Sequence events across a conversation, deduplicate within a turn, and replay only inside the authorized workspace/principal boundary.
- Store cancellation requests durably now. Add actual Stop enforcement and a cancelled terminal state with streaming in Step 23.
- Never persist credentials or hidden chain-of-thought in event payloads.

## Consequences

Later capabilities share one policy and event surface. Existing non-streaming behavior remains stable. The first implementation supports cursor polling, not SSE; definitions contain schemas only and cannot execute code. SQLite remains appropriate for local mode but must be replaced or adapted for multi-host deployment.

## Rejected alternatives

- A separate execution path for each connector/plugin: inconsistent policy and revocation.
- Letting skill instructions grant tools: instruction text is not authority.
- Storing provider-native stream events directly: leaks provider coupling and weakens replay compatibility.
- Adding coding/subagent execution first: expands host risk before approval, audit, and cancellation contracts exist.
