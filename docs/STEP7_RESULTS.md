# Step 7 results — retrieval identity and reranker lifecycle

27 October 2026. The local Step 7 acceptance cases pass with synthetic offline data.

## Built

- Retrieval now tokenizes case and punctuation variants consistently while retaining whole domain identifiers such as `INV-2048` and `v2.1`. BM25 only contributes documents with at least one matching token. A query with no vector hit and no lexical overlap produces no evidence and an explicit `insufficient_evidence` answer status.
- Reciprocal rank fusion uses ingestion's stable `chunk_id`; equal text from different chunks and sources stays separate. The browser source cards no longer collapse distinct returned chunks.
- BM25 indexes and token sets are cached only by authorized workspace, immutable corpus version, and tokenizer version. The cache is limited to four indexes; vectorstores are not retained in a process-global search cache. `clear_retrieval_caches()` explicitly invalidates it.
- Reranker loading is synchronized and keyed by model configuration. A failed load is retried after five seconds or after explicit cache clear; successful models are limited to two cached configurations. Candidate count and pair lengths are bounded. Wrong count, nonnumeric, or nonfinite scores trigger deterministic retrieval-order fallback.
- API responses and stored turns include candidate counts, raw scores when available, model identity, status, and a safe degradation reason. The UI shows when reranking falls back, including after a conversation reload. Raw scores are ranking values, not probabilities.

## Validation

- `python3 -m unittest discover -s tests -t . -q`: 116 offline tests pass.
- `python3 -m tools.audit_failure_modes --output /private/tmp/docqa-step7-diagnostics.json`: 2 reproduced limitations (M03, A05), 23 not reproduced, 1 control passed, 0 probe errors. R01–R04 no longer reproduce.
- The same candidate pair was checked with reranking enabled and disabled. A mocked model reversed their order; fallback retained fusion order. Transient load recovery, malformed score handling, identifier matching, provenance, cache eviction, and empty evidence have focused tests.
- Synthetic browser preview on `127.0.0.1:8773`: the answer showed the reranker fallback notice, which persisted after reload.
- `python3 -m compileall -q agents app langgraph_flow retrieval tests tools` and `git diff --check` pass.

## Limits and next work

No relevance threshold for FAISS hits is calibrated yet, so a weak vector hit may still reach the answer model. The current no-evidence abstention covers empty vector results and zero lexical overlap; confidence and citation support require labeled evaluation. Real reranker quality, latency, and model-loading behavior on target hardware were not measured. The fallback retains fused order and reports degradation instead of presenting a score as confidence.

Next: Step 8 should remove or apply the unused response plan, label evidence with stable IDs in final answers, verify citations and source versions, and bound retries. In parallel, finish Step 3 OCR/resource measurements and Step 6 retention and multi-worker semantics before shared deployment.
