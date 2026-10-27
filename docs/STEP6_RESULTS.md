# Step 6 progress — durable conversations

27 October 2026. This is the first Step 6 slice, not the full acceptance gate.

## Implemented

- SQLite owns conversation titles, completed turns, and idempotency keys under a server-authorized workspace and principal. The browser sends only the current question, conversation ID, and request key; supplied `chat_history` is ignored. Conversation list/detail endpoints reopen the transcript after reload or API restart.
- Duplicate requests replay the stored answer, including when the response to a newly created conversation was lost. A conflicting in-flight turn returns 409. Startup marks turns interrupted by the previous single-worker process as failed so the same request key can retry.
- The graph receives at most eight completed turns and 16,000 characters of recent context. Referential follow-ups append the prior user question to retrieval/web search. The prompt includes history even without a profile. The disconnected memory helper returns copies of nested messages.
- Graph invocation and corpus snapshot loading run in a worker thread. Two requests may execute concurrently per API process; further requests receive 503 after one second waiting. The UI can create, reopen, and switch conversations, including between workspaces. Answers from a replaced corpus are marked stale.

## Verification

- `python3 -m unittest discover -s tests -t . -q`: 110 offline tests pass.
- `python3 -m tools.audit_failure_modes --output /private/tmp/docqa-conversations-diagnostics.json`: 6 reproduced limitations, 19 not reproduced, 1 control passed, 0 probe errors. M01, M04, and P02 no longer reproduce. M03 remains because the legacy `get_memory()` factory is still ephemeral; the active API uses SQLite.
- Synthetic preview on `127.0.0.1:8772`: a saved answer reappeared after reload; New chat cleared it; switching from alpha to beta showed no alpha conversation. This used synthetic corpora/model output.
- `python3 -m compileall -q app agents memory storage tests tools` and `git diff --check` pass.

## Remaining gate and limits

Graph state is transient because it currently includes FAISS and Pandas objects. SQLite is the authoritative transcript, not a LangGraph checkpoint. A future serializable graph state can use a durable checkpointer without creating a second transcript. The current process recovery assumes one API worker; a multi-worker deployment needs leased turns and a shared database. History is bounded for prompts but older turns have no summary or retention/deletion policy. Referential query expansion is conservative and does not clarify ambiguous references. The browser conversation list shows 20 recent threads, and detail reads all completed turns without pagination. Worker threads cannot forcibly cancel a model call; full job deadlines, streaming status, and load tests remain Step 10 work.

Next: Step 7 provenance-preserving retrieval and visible reranker lifecycle, alongside Step 3 resource/OCR measurements. Complete Step 6 summarization, unresolved-reference handling, retention policy, and multi-worker semantics before claiming its full gate.
