# Step 12 results — label-blind Ollama evaluation and review handoff

Verified 28 October 2026. The evaluation pipeline is executable end to end. Release and fine-tuning remain blocked; no production-readiness claim is made.

## Implemented

- Added separate inference manifests and label files. The runner rejects answer, rubric, expected-fragment, target, and manual-review fields anywhere in its input.
- Frozen source bytes carry individual SHA-256 digests. The run and labels must match one corpus-bundle digest and dataset ID.
- Added a real-graph Ollama runner covering planning, hybrid retrieval, deterministic table execution, answer generation, evidence validation, latency, per-call tokens, model revision, and embedding identity.
- Added opt-in model-call telemetry that records provider/model, duration, and token counts without storing prompts or responses.
- Added manual-review worksheet generation and strict attachment. Release requires a named reviewer, timestamp, human method, explicit independence, and complete boolean decisions for every case.
- Added a dataset lifecycle control: an exposed set can be marked `release_candidate: false`, which prevents release eligibility even if its metrics later pass.
- Added deterministic copy-only migration evidence generation so the combined gate can be executed from scratch.
- Hardened fallback retrieval when the optional cross-encoder is unavailable: meaningful-token overlap, relative BM25 filtering, calibrated MiniLM L2 rejection, explicit cross-language state, and explicit document-version filtering.
- Standardized insufficient-evidence responses and preserved all reviewed evidence when generation supplies no citations.

## Real-model evidence

Candidate: unfine-tuned local Ollama `llama3.2:3b`, revision `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`. Embedding identity: 384 dimensions, fingerprint `437779813d35cbe929ed6c3f29116915da82b6cdda8fca6d48c60e975889e547`.

The initial set became calibration data after its results drove retrieval fixes. Its final calibration run reached full automated targets: answer contract 0.75, Recall@5 1.0, MRR 1.0, abstention 1.0, citation precision/recall 1.0, clean unanswerable rate 1.0, table exactness 1.0, p95 2,982 ms, and 4,538 input / 645 output tokens. It is permanently non-release.

A fresh one-shot release attempt then measured:

- coverage 1.0; answer contract 0.875; abstention 1.0;
- Recall@1 0.714, Recall@3/5 0.857, MRR 0.857;
- citation precision 0.786 and recall 0.786;
- clean unanswerable rate 1.0 and table exactness 1.0;
- p50 2,007 ms, p95 6,721 ms; 4,563 input / 884 output tokens.

It failed the 0.80 citation-precision target. The run exposed an irrelevant temporal source, incomplete duplicate provenance, and wrong-version follow-up retrieval. Those failures were not hidden: the set was retired before the resulting fixes, so it cannot be reused as release evidence.

## Verification

- Focused label-isolation, review, scorer, provider, retrieval, API, graph, and evidence tests pass.
- Full offline baseline passes 178 tests after the final follow-up hardening.
- Historical diagnostics remain zero reproduced defects, 25 not reproduced, one rejection control passed, zero control failures, and zero probe errors.
- Synthetic two-version production-storage rollback evidence passes all five checks without source mutation.
- Combined release gate correctly fails only when held-out model evaluation is not release eligible.

## Remaining blockers

1. Freeze a new, never-executed release set after the latest retrieval fixes. Run the candidate once; do not alter code or labels based on it and call the same run held out.
2. If automated targets pass, send the generated worksheet to an independent human reviewer. This agent cannot truthfully self-certify reviewer independence.
3. Attach the completed review, rescore with `--require-release-eligible`, and rerun the combined evidence gate.
4. Cache and validate the configured cross-encoder or explicitly accept the measured fallback. Current relevance distance is calibrated only for the recorded MiniLM embedding identity.
5. Run deployment-specific multi-process/load, trusted identity, physical-device UI, and retention approval before production.

Fine-tuning is not justified yet. The failed fresh run identified retrieval/reference resolution gaps, not a demonstrated model-only behavior gap.
