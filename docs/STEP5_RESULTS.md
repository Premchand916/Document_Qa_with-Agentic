# Step 5 — Validated table routing and calculations

Verified 27 October 2026. Step 5's stated deterministic acceptance cases pass offline. Checkpoint B remains open until Step 3 resource isolation and Step 4 real-model/rollback checks finish.

## Behavior

- `agents/planner_agent.py` routes by question and server-owned sources. A PDF question uses retrieval even when a CSV is present; an LLM intent label cannot force an unrelated table route.
- `agents/table_plan.py` produces a typed, deterministic plan for one source/sheet and `sum`, `mean`, `min`, `max`, `count`, row count, or schema. It resolves an explicit filename or sheet, then checks columns and one typed equality filter. Unknown columns, ambiguous sources/filter values, and unsupported operations return clarification.
- `agents/dataframe_tool.py` rebuilds and validates the plan against authorized assets before executing fixed Pandas operations. Results include exact value, source, sheet, selected row numbers, filters, and missing-value policy. The API exposes `result_status` and `tool_result`; the visible answer preserves calculated values without another model call.

## Validation

The full offline suite passes **97 tests**. Direct and real API/graph tests verify second-table total **99**, North-only total **10**, nullable total **40**, PDF routing with a CSV present, ambiguity, unknown columns, unsupported operations, forged plans, named sheet selection, and each allowlisted operation. The diagnostic harness reports **9 reproduced limitations**, **16 not reproduced**, one passing rejection control, and zero probe errors. A01–A03 changed to not reproduced; A06 was already repaired.

## Limits

The query parser supports one equality filter and clear source/column names. Multiple filters, grouping, joins, derived metrics, date comparisons, and fuzzy natural-language references require new contracts and fixtures; they produce clarification rather than a guessed result. LLM intent and response planning still run before deterministic routing, adding avoidable latency. Live-model accuracy and production concurrency are not measured. Step 6 must add durable, server-owned conversations before follow-up table requests can rely on prior context.
