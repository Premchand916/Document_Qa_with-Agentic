# Step 23 Results — Streaming Turns and Stop

**Status:** implemented locally and verified offline.  
**Date:** 5 November 2026

## Behavior

- `POST /api/chat/turns` reserves an idempotent, owner-scoped turn and returns its conversation and turn IDs plus an attempt cursor with HTTP 202. A bounded background task executes the existing chat workflow. The attempt cursor excludes terminal events from a previous retry with the same key. `POST /api/chat` remains available for synchronous clients.
- `GET /api/conversations/{conversation_id}/events/stream?turn_id=...` sends durable SSE events with numeric IDs. `Last-Event-ID` resumes after the last delivered sequence; the browser also discards duplicate sequences. The stream rechecks connector access and redacts revoked-source answers before each replay batch.
- The stream has 30-second sessions, 100-event batches, a 500-event session cap, and heartbeat comments. The browser reconnects up to four times and displays generation, validation, and tool activity. The validated final answer comes from the existing conversation API.
- `DELETE /api/conversations/{conversation_id}/turns/{turn_id}` atomically changes a running turn to `cancelled`, stores a visible “Stopped.” response, and appends request and terminal events. A later model result cannot complete the cancelled turn. Repeated Stop requests return the same durable cancellation state.
- Existing SQLite turn tables migrate to the new terminal status in place. Conversation history and summaries still use completed answers only; the transcript can display stopped turns.

## Verification

- Focused event tests: 11 pass, covering SSE cursor replay, invalid cursors, owner scope, migration, asynchronous completion, idempotent Stop, and a Stop race against a blocked graph result.
- Offline baseline: 235 tests and all seven stages pass.
- Python compile check, inline JavaScript parse, and `git diff --check` pass.
- Local Chrome rendered the updated chat shell. An isolated temporary-data server with a deliberately slow synthetic model showed “Generating response…”, an enabled Stop control, and a persisted “Stopped.” message after clicking Stop. No real model or existing conversation was used for that interaction.

## Limits and next action

Progress is phase and tool status, followed by a validated final message. The current graph returns one answer, so true token deltas would require a provider streaming adapter plus buffering/verification before citation-bearing text becomes visible. Stop makes the persisted turn terminal, but Python cannot forcibly interrupt a model call already running in a worker thread; the existing deadline and chat-slot limit bound that work. In-process background tasks and SQLite remain single-host assumptions. The next capability step is Step 24's policy-filtered tool loop; token streaming can be added after a verified output protocol is defined.
