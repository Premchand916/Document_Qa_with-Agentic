# Issue draft: enforce request and rendering boundaries

Local draft only; not published. Prepared for the repository's major-change issue workflow.

Blank queries and malformed history reach application/model code, upload work has no byte/count ceiling, and generated Markdown can enter the DOM unsanitized. There is no established HTTP identity boundary. Add strict request schemas, streamed request/file limits, safe errors, explicit local access, a trusted authentication-proxy adapter, and sanitized rendering.

Acceptance evidence: 50 offline tests pass, three audited failures no longer reproduce, and 14 browser sanitizer fixtures pass. See [Step 1 results](../STEP1_RESULTS.md) for the integration contract, tests, and limits.

Follow-ups: Step 2 must replace global corpus/provider state before multi-workspace claims. Shared deployment still needs a configured identity provider and validated proxy setup. A mobile layout issue was observed at 393 × 852: the fixed sidebars squeeze the chat panel out of view; track a separate responsive-layout change.
