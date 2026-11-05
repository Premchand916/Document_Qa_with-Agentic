# Step 24 Results — Bounded Tool Loop and Preference Approval

**Status:** first local tool-loop slice implemented and verified. Grounded document/web routing remains a separate integration gate.  
**Date:** 5 November 2026

## What changed

- Added opt-in `use_tools` to chat requests and a visible **Use built-in tools in general chat** toggle. Existing requests continue through the original chat path.
- Registered typed `math.evaluate` and `time.current_utc` read capabilities, plus `memory.save_preference` as a write proposal. The server intersects contracts with capability policy and a hardcoded executor allowlist before sending them to the model. Model output is one strict JSON action; unknown tools, extra fields, invalid arguments, and unsupported code fail closed.
- The loop caps work at four model steps, three completed tool calls, 8,192 estimated tokens, a 12,000-character prompt, a 4,000-character model action, and 30 seconds checked between calls. The existing chat deadline and slot limit remain outer bounds. There are no automatic retries.
- Arithmetic uses a bounded AST and decimal math; it never runs generated code or shell commands. Tool results are shown separately from the model's final text. Turn events expose selections, requests, starts, completions, and safe failure reasons without raw arguments or hidden reasoning.
- A model may propose a preference write during an active leased turn. The proposal is owner-scoped, secret-checked, capped at 20 pending requests, and expires after seven days. It cannot write memory until the user explicitly approves. Approval or rejection updates the transcript and event log in the same SQLite transaction as the semantic-memory write. Repeated decisions are idempotent; Stop, failed turns, lease expiry, and conversation deletion discard pending proposals.
- The chat card displays the exact proposed key and value with Approve and Reject controls. Approval immediately refreshes the Memory panel count. Current approval state survives a page refresh.

## API

- `POST /api/chat` or `POST /api/chat/turns` with `"use_tools": true` for general chat.
- `GET /api/tool-approvals/{approval_id}` returns only the authenticated owner's proposal.
- `POST /api/tool-approvals/{approval_id}/decision` with `{"decision":"approve"}` or `{"decision":"reject"}` resolves it once.

## Verification

- Eleven focused tool-loop tests cover deterministic arithmetic, code/exponent rejection, policy denial, malformed calls, call and context budgets, opt-in compatibility, approval isolation, secret rejection, expiry, capacity failure, Stop cleanup, and duplicate decisions.
- Full offline baseline: 246 tests and all seven stages pass. Python compilation, inline JavaScript parsing, and `git diff --check` pass.
- In isolated Chrome with a synthetic model and temporary data, the general-chat toggle produced an approval card. Clicking Approve updated the card to approved and the Memory navigation count from zero to one. No real model, user data, or external service was used for this browser check.

## Boundaries and next gate

The tool loop is intentionally limited to general chat. Grounded document and web answers still use the existing validated graph, and the toggle does not redirect those turns through model-selected tools. Provider token counts are used when present; otherwise the cap is a character estimate. The 30-second check cannot interrupt a synchronous provider call already in progress; the outer turn deadline and slot limit contain that work. Only preference writes have an approval executor. Remote tools, arbitrary plugins, coding/shell execution, consequential actions, and multi-host approvals are outside this slice.

Next: integrate the same tool contracts with grounded document/web evidence without letting a model bypass citations, then evaluate tool-selection quality on local Ollama before enabling more capabilities.
