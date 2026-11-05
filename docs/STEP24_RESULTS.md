# Step 24 Results — Bounded Tool Loop and Preference Approval

**Status:** local tool loop and evidence-bound grounded calculations implemented and verified; broad tool selection remains an evaluation gate.
**Date:** 5 November 2026

## What changed

- Added opt-in `use_tools` to chat requests and a visible **Use built-in tools** toggle. Existing requests continue through the original chat path.
- Registered typed `math.evaluate` and `time.current_utc` read capabilities, plus `memory.save_preference` as a write proposal. The server intersects contracts with capability policy and a hardcoded executor allowlist before sending them to the model. Model output is one strict JSON action; unknown tools, extra fields, invalid arguments, and unsupported code fail closed.
- The loop caps work at four model steps, three completed tool calls, 8,192 estimated tokens, a 12,000-character prompt, a 4,000-character model action, and 30 seconds checked between calls. The existing chat deadline and slot limit remain outer bounds. There are no automatic retries.
- Arithmetic uses a bounded AST and decimal math; it never runs generated code or shell commands. Tool results are shown separately from the model's final text. Turn events expose selections, requests, starts, completions, and safe failure reasons without raw arguments or hidden reasoning.
- A model may propose a preference write during an active leased turn. The proposal is owner-scoped, secret-checked, capped at 20 pending requests, and expires after seven days. It cannot write memory until the user explicitly approves. Approval or rejection updates the transcript and event log in the same SQLite transaction as the semantic-memory write. Repeated decisions are idempotent; Stop, failed turns, lease expiry, and conversation deletion discard pending proposals.
- The chat card displays the exact proposed key and value with Approve and Reject controls. Approval immediately refreshes the Memory panel count. Current approval state survives a page refresh.
- Document and web retrieval now pass through a read-only grounded-calculation node before the existing QA and citation validator. It allows `math.evaluate` only when the user's question asks for arithmetic. Each operand must occur in the bounded evidence excerpt sent to the planner; results carry matching evidence IDs. The planner's final text is discarded. The QA agent receives validated results; if its answer uses the calculated value with a different citation, validation rejects the answer and hides the tool result. Connector freshness, ACL, epoch checks, and stored-answer redaction still run on this path.
- Ollama tool-planning calls request JSON output format. Ordinary model responses keep their previous format. The server still validates action shape, capability policy, arguments, and evidence binding. A rejected calculation records a safe tool failure and QA continues from available sources.

## API

- `POST /api/chat` or `POST /api/chat/turns` with `"use_tools": true` for general chat, or for evidence-bound arithmetic in document/web chat.
- `GET /api/tool-approvals/{approval_id}` returns only the authenticated owner's proposal.
- `POST /api/tool-approvals/{approval_id}/decision` with `{"decision":"approve"}` or `{"decision":"reject"}` resolves it once.

## Verification

- Eighteen focused tool-loop tests cover deterministic arithmetic, code/exponent rejection, policy denial, malformed calls, call and context budgets, opt-in compatibility, approval isolation, secret rejection, expiry, capacity failure, Stop cleanup, duplicate decisions, grounded document/web citations, wrong-source citation rejection, connector revocation redaction, irrelevant-query gating, and Ollama JSON-mode isolation.
- Full offline baseline: 253 tests and all seven stages pass. Python compilation, inline JavaScript parsing, and `git diff --check` pass.
- In isolated Chrome with a synthetic model and temporary data, the general-chat toggle produced an approval card. Clicking Approve updated the card to approved and the Memory navigation count from zero to one. No real model, user data, or external service was used for this browser check.
- A second isolated Chrome pass used a synthetic published document and model: enabling built-in tools, asking for the combined deadline, and submitting the turn showed a 45-day result card, its evidence ID, the source card, and a matching answer citation. The server and temporary data were removed after the check.
- Local Ollama `llama3.2:3b` passed four synthetic grounded tool-selection checks after JSON-mode and calculation-intent changes: relevant arithmetic, irrelevant summary, hostile evidence text, and a missing operand. This is a narrow smoke check, not a representative quality score.

## Boundaries and next gate

Grounded document/web mode permits one read-only calculation from at most six evidence excerpts of 700 characters each; it does not expose preference writes, UTC time, or arbitrary source-fetching tools. Numeric presence proves operand provenance, not that the chosen operation is semantically correct; the QA answer and citations still require review. Provider token counts are used when present; otherwise the cap is a character estimate. The 30-second check cannot interrupt a synchronous provider call already in progress; the outer turn deadline and slot limit contain that work. Only preference writes have an approval executor. Remote tools, arbitrary plugins, coding/shell execution, consequential actions, and multi-host approvals are outside this slice.

Next: build a representative held-out Ollama tool-selection set and measure operation choice, abstention, latency, and citation quality before Step 25's plugin/MCP gateway. Keep fine-tuning paused.
