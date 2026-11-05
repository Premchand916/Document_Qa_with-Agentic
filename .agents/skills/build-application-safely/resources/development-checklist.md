# Application Development Checklist

Use for feature work, fixes, architecture changes, integrations, migrations, and release preparation. Mark each applicable item. Write `N/A — reason` when an item does not apply.

## 0. Task checkpoint

- [ ] User outcome stated in one sentence.
- [ ] In-scope and out-of-scope behavior recorded.
- [ ] Repository guidance and steering/checkpoint read.
- [ ] Existing dirty changes identified and preserved.
- [ ] Current behavior reproduced or inspected.
- [ ] Completion gate defined with observable evidence.
- [ ] Exact next action recorded.

## 1. Authority and safety

- [ ] Requested work authorizes every planned mutation.
- [ ] Destructive or irreversible targets resolved exactly.
- [ ] Secrets, credentials, private data, and production data excluded from output/logs.
- [ ] External writes, messages, deployments, installs, commits, and pushes have explicit authorization.
- [ ] Identity and permissions remain server-owned; client input grants no authority.
- [ ] Failure behavior is fail-closed for security, ownership, integrity, and approval checks.

## 2. Desired behavior

- [ ] Happy path described from user action to durable result.
- [ ] Empty, loading, disabled, retry, partial, timeout, cancellation, and error states considered.
- [ ] Inputs, outputs, schemas, status codes, and compatibility expectations defined.
- [ ] Evidence, freshness, provenance, and citation behavior defined when facts are returned.
- [ ] One desired-behavior test fails for the correct reason before the fix when practical.

## 3. Breakpoint map

Score relevant risks by impact, likelihood, and detectability. Address highest combined risk first.

### Contracts and input

- [ ] Missing, null, blank, malformed, oversized, duplicate, ambiguous, and hostile values.
- [ ] Encoding, Unicode, file type, content type, and boundary values.
- [ ] Unknown fields, incompatible versions, and unsafe defaults.

### Identity and authorization

- [ ] Forged identity, workspace/tenant crossing, horizontal and vertical privilege escalation.
- [ ] Missing scopes, excessive scopes, approval bypass, revocation, and deletion propagation.
- [ ] Resource existence does not leak across authorization boundaries.

### State and storage

- [ ] Partial writes, transaction boundaries, failed publication, disk/database errors.
- [ ] Schema migration, downgrade/rollback, corrupted data, stale caches, and incompatible artifacts.
- [ ] Restart recovery, retention, deletion cascade, backup, and restore.

### Concurrency and lifecycle

- [ ] Duplicate requests, idempotency-key conflicts, races, stale workers, and lease expiry.
- [ ] Timeouts, retries, cancellation, backpressure, queue saturation, and retry multiplication.
- [ ] Exactly one durable terminal outcome.

### External dependencies

- [ ] Provider unavailable, slow, rate-limited, revoked, schema-changed, or partially successful.
- [ ] Stale remote data, cursor replay, webhook duplication, and network interruption.
- [ ] Credentials remain external references and never enter prompts, events, artifacts, or logs.

### Resource bounds

- [ ] File count/bytes, rows/pages, context/tokens, recursion/depth, output size, workers, memory, and runtime bounded.
- [ ] Slow clients and abandoned work cannot consume capacity indefinitely.
- [ ] Cache keys include ownership, immutable version, configuration, and query inputs.

### AI-specific boundaries

- [ ] Retrieved content, memory, skill text, web pages, and connector data treated as untrusted.
- [ ] Prompt injection cannot grant tools, permissions, secrets, or instruction priority.
- [ ] Unsupported claims abstain or clarify; calculations remain deterministic where possible.
- [ ] Tool schemas, budgets, idempotency, approvals, evidence, and errors remain visible.
- [ ] Hidden reasoning is never required, stored, or exposed.

### UI and accessibility

- [ ] Primary interaction, keyboard path, focus, labels, contrast, responsive layout, and overflow checked.
- [ ] Loading, disabled, empty, error, retry, stale, offline, and permission-denied states visible.
- [ ] Unsafe HTML/Markdown/URLs cannot execute.
- [ ] UI state matches durable server state after refresh and workspace switching.

### Operations

- [ ] Safe error codes and request/job IDs available without sensitive detail.
- [ ] Metrics cover latency, capacity, errors, degradation, queue time, and provider state.
- [ ] Audit events contain safe observable actions, not secrets or hidden reasoning.
- [ ] Rollback, cleanup, retention, and ownership responsibilities documented.

## 4. Implementation plan

- [ ] Smallest coherent vertical slice selected.
- [ ] Existing architecture, naming, repositories, and dependency choices reused.
- [ ] Files and interfaces to change listed.
- [ ] Migration and backward-compatibility path defined.
- [ ] Test plan maps each high-risk breakpoint to observable evidence.
- [ ] Unrelated refactors and speculative features excluded.

## 5. Implementation checks

- [ ] Validation occurs before expensive or external work.
- [ ] Ownership and policy checks occur before data access or capability exposure.
- [ ] Durable state and lifecycle events commit atomically where required.
- [ ] Retries are limited to safe/idempotent operations.
- [ ] Errors are actionable to users and redacted externally.
- [ ] Payload, context, concurrency, runtime, and retention limits enforced.
- [ ] New configuration has safe parsing, bounds, defaults, and startup failure behavior.
- [ ] No credentials, generated data, indexes, or private artifacts added to version control.

## 6. Verification ladder

- [ ] Focused unit/contract tests pass.
- [ ] Adversarial and negative tests pass.
- [ ] Ownership and cross-tenant tests pass.
- [ ] Persistence, restart, idempotency, deletion, and failure-atomicity tests pass where relevant.
- [ ] Concurrency, timeout, retry, cancellation, and capacity tests pass where relevant.
- [ ] API integration tests pass with real routing/storage boundaries.
- [ ] UI interaction, responsive layout, accessibility, and error states inspected for UI changes.
- [ ] Compile, schema, static analysis, formatting, and diff checks pass.
- [ ] Impacted regression suites pass.
- [ ] Full repository baseline passes at milestone.
- [ ] Real provider/model/browser/connector smoke test performed when available.
- [ ] Verified, simulated, inferred, and unverified claims separated.

## 7. Failure repair loop

- [ ] Failure reproduced deterministically.
- [ ] Raw evidence preserved without secrets.
- [ ] Smallest failing boundary isolated.
- [ ] Root cause distinguished from symptom.
- [ ] Regression test added or strengthened.
- [ ] Narrow fix implemented without unrelated behavior change.
- [ ] Focused test rerun, then impacted suite, then full gate.
- [ ] Checkpoint and limitations updated.

Never hide a failure using unlimited retries, broad exception swallowing, weaker assertions, changed labels, blanket skips, or unreported fallback.

## 8. Completion and handoff

- [ ] Acceptance behavior demonstrated end to end.
- [ ] Required security, ownership, integrity, and evidence gates pass.
- [ ] Material limitations and deliberate non-goals documented.
- [ ] Changed files and important lines recorded.
- [ ] Verification commands and results recorded.
- [ ] Rollback/recovery instructions recorded when state or deployment changed.
- [ ] Steering/checkpoint updated with completed and remaining work.
- [ ] One exact next action recorded.
- [ ] Commit/push/deploy commands provided only when appropriate; action performed only with authorization.

## Handoff template

```text
Objective:
Constraints:
Completed:
Files changed:
Verification:
Verified behavior:
Simulated behavior:
Limitations/blockers:
Rollback/recovery:
Exact next action:
```
