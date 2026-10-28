# Step 8 results — one plan and validated evidence

28 October 2026. Local Step 8 acceptance cases pass with synthetic offline inputs.

## Built

- Active graph now uses one response-planning model call before deterministic document/table routing. Document answers consume the bounded plan; table execution remains deterministic and web search bypasses document planning. The disconnected intent-model node is removed from the active graph.
- Every returned document, table result, and web result receives a repeatable `E-<hash>` evidence ID. Local evidence records carry the pinned corpus version; web evidence records retain web origin and URL metadata.
- Generated citation IDs are checked against the exact evidence returned for that response. Unknown IDs fail closed with `citation_error`. When a model omits citations, the system adds a clearly labeled `Evidence reviewed` reference to an existing record rather than inventing an ID.
- Differing explicit values across multiple sources produce a `possible_conflict` signal. Both sources stay visible and the QA prompt tells the model to show the disagreement. This is a conservative signal, not semantic conflict resolution.
- Missing document storage routes to an actionable `insufficient_evidence` result instead of dereferencing a null vector store. Web configuration/search failures return `tool_error`; empty results return `insufficient_evidence`.
- API responses, durable conversation turns, and UI reloads preserve citations, evidence status, corpus version, retrieval status, and exact table results. Unscoped legacy process memory is retired; the authorized SQLite conversation repository is the only active transcript owner.

## Validation

- `python3 -m unittest discover -s tests -t . -q`: 123 tests pass.
- `python3 -m tools.audit_failure_modes --output /private/tmp/docqa-step8-diagnostics.json`: 0 reproduced defects, 25 not reproduced, 1 rejection control passed, 0 probe errors.
- `python3 -m tools.run_baseline --output-dir /private/tmp/docqa-foundation-from-scratch`: fresh subprocesses passed tests, diagnostics, environment inspection, and the zero-defect diagnostic gate. The runner now fails on any reproduced defect, failed control, or probe error. This recreates checks from clean temporary application state; it does not reinstall dependencies or invoke live models.
- Workflow traces show document `entry → response_planner → planner → retriever → qa_agent → finalize` with two model calls; table `entry → response_planner → planner → dataframe_tool → finalize` with one; web `entry → web_search → qa_agent → finalize` with one.
- Focused contracts cover plan consumption, stable evidence IDs, invalid citations, source conflicts, null stores, exact table results, web origin, and legacy-memory retirement.
- Synthetic browser preview on `127.0.0.1:8774` showed a resolved citation and reranker fallback notice; both survived reload.
- `python3 -m compileall -q agents app embeddings ingestion langgraph_flow memory retrieval storage tests tools utils vector_Store` and `git diff --check` pass.

## Limits and next work

Citation validation proves that IDs exist in current evidence; it does not prove every generated claim is semantically supported. Conflict detection compares explicit numeric/date-like values and may miss textual contradictions or flag unrelated values. FAISS sufficiency still needs calibration on held-out data. Live Tavily, Gemini, and Ollama quality/latency were not tested in this offline gate.

Next foundation work: finish Step 3 OCR/resource measurements and Step 6 retention/multi-worker semantics, then build Step 9 opt-in episodic feedback. Do not train on chat history automatically. Step 10 still owns durable jobs, cancellation, stage metrics, and load tests.
