# Step 11 results — evaluation, migration, and index decisions

Verified 28 October 2026. Local evidence tooling passes. Release and fine-tuning readiness remain blocked because no separately frozen held-out real-model run has been supplied.

## Implemented

- Evaluation schema now distinguishes `development` from `holdout`. Development fixtures cannot produce `release_eligible: true`.
- Rich run envelope requires provider, model, revision, embedding identity, timestamp, split, and corpus fixture hash.
- Scorer measures evidence Recall@1/3/5, MRR, source citation precision/recall, literal answer contracts, abstention accuracy, unanswerable evidence cleanliness, exact typed table results, p50/p95 latency, tokens, cost, and independent manual semantic/citation review.
- Declared quality targets produce a deterministic gate. Missing cases, malformed runs, label/run split mismatch, incomplete manual review, or development labels block release eligibility.
- Copy-only migration rehearsal uses SQLite backup plus immutable artifact copies. It loads current version, rolls back, reconstructs repository, restores current version, reconstructs again, and verifies source version/checksum immutability.
- Workspace storage exposes ordered version metadata for migration tooling. No public rollback endpoint was added because trusted administrator roles do not exist yet.
- Isolated FAISS benchmark compares exact `IndexFlatL2` with `IndexHNSWFlat`, recording build/search time, serialized size, RSS, Recall@k, and a non-mutating recommendation.
- Combined release-evidence tool requires passing baseline, held-out evaluation, migration rehearsal, and index decision evidence.

## Verification

- Full baseline passes **169 tests**.
- Historical diagnostics: zero reproduced defects, 25 not reproduced, one passing rejection control, zero control failures, and zero probe errors.
- Real local ingestion/OCR, concurrent parsing, cached embeddings, environment inspection, and isolated index benchmark pass.
- Copy-only rollback tests restore previous/current versions across repository reconstruction, preserve source fingerprint, reject insufficient history, and refuse non-empty copy targets.
- Evaluation tests cover rank movement, citation hallucination, missing citation recall, typed table mismatch, unanswerable evidence leakage, rich metadata, split mismatch, quality targets, and release eligibility.

## Index decision

Synthetic CPU measurement: 5,000 vectors, 64 dimensions, 100 queries, Recall@10, one thread.

- Flat: exact; p95 0.0203 ms; build 0.108 ms; 1,280,045 serialized bytes.
- HNSW (`M=32`, `efConstruction=80`, `efSearch=64`): Recall@10 0.981; p95 0.0395 ms; build 163.067 ms; 2,637,250 serialized bytes.
- HNSW p95 speedup: 0.515×, meaning it was slower in this small synthetic workload.
- Decision: **retain Flat**. Exact Flat is well below the declared 5 ms microbenchmark target; production index remains unchanged.

This follows FAISS guidance that Flat is the exact baseline and is appropriate when exact results are required or search volume/scale does not amortize more complex index construction. HNSW remains a future option when measured scale makes Flat miss latency targets and HNSW meets recall/speedup requirements. See [FAISS index selection guidance](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) and [FAISS index reference](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).

## Remaining blockers

- Current 11-case fixture is visible development data, not held-out evidence.
- No real Ollama/Gemini candidate run, token/latency trace, or independent manual review was executed in this step.
- Microbenchmark is synthetic vector search, not end-to-end hybrid retrieval or answer latency at 10,000+ production chunks.
- Migration rehearsal is offline and single-host. Online migration, active writers, multi-host coordination, and deployment rollback approval remain open.
- Fine-tuning should not start until held-out evaluation identifies a model-behavior gap rather than retrieval, evidence, or tool failure.

Next: freeze unseen task labels, run local Ollama candidate and unfine-tuned baseline, complete independent review, then execute `tools.verify_release_evidence`.
