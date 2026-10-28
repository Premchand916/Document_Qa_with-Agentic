# Offline Baseline — Step 0

Verified 26 October 2026. Production application behavior and `requirements.txt` are unchanged.

## Results and meaning

- **28 unittest tests passed**, with no skips or expected failures. They protect working ingestion, document/table/web graph paths, API rejection controls, FAISS persistence, reranking, and evaluation bookkeeping.
- **25 failure/limitation probes reproduced; one rejection control passed; zero probe errors.** These remain unresolved. A diagnostic exit code of zero means the probes executed, not that the application is correct.
- **11 labeled evaluation cases** cover PDF QA, complete/filtered/nullable table calculations, table selection, follow-ups, missing answers, duplicate sources, conflicting versions, Spanish, and hostile instructions.
- **26 audit IDs** have a future implementation-step owner and desired behavior in `tests/fixtures/audit_ownership.json`. The ownership check reads probe registration without running those failures as passing tests.
- **117 installed packages** in the runtime/test dependency closure were captured; no missing or incompatible declared dependencies were found for this environment.

Retained evidence: [summary](baseline/summary.json), [diagnostics](baseline/diagnostics.json), [environment](baseline/environment.json), and [dependency constraints](baseline/constraints.txt). The original `failure-mode-evidence.json` remains untouched. Fixture hashes in the summary identify the exact labeled inputs used.

## Run locally

From the repository root, with project dependencies installed:

```bash
python3 -m tools.run_baseline
```

This runs tests, diagnostics, real local ingestion/OCR plus cached-embedding measurement, and environment inspection in separate subprocesses. Fresh logs and JSON go to ignored `.baseline-results/`. Since Step 8 repaired all registered probes, any reproduced defect, failed control, probe error, failed ingestion fragment check, embedding failure, dependency inconsistency, or stage timeout now produces a nonzero exit. Tesseract and the locally cached embedding model are required for the measurement. Historical Step 0 results above remain the dated baseline.

Focused checks:

```bash
python3 -m unittest discover -s tests -t . -v
python3 -m unittest tests.test_retrieval -v
python3 -m unittest tests.test_api_and_graph -v
```

No new test-runner dependency is needed: `unittest` ships with Python. `requirements-dev.txt` adds the HTTPX dependency used by FastAPI TestClient. For a candidate recreation on the recorded platform:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt -c docs/baseline/constraints.txt
.venv/bin/python -m tools.run_baseline
```

Installation may require package downloads. This recipe was **not** tested in a fresh environment. Constraints pin the installed dependency closure for macOS arm64 / CPython 3.14.6; they are not a cross-platform or hash-verified lock. Python 3.14.6 is the verified baseline version, not a claim about every supported Python version. Model weights and external OCR executables are outside this snapshot.

## Isolation and native-library behavior

Application tests use synthetic inputs, temporary vector directories, explicit application state, deterministic two-dimensional embedding fixtures, and stubbed model/search responses. They exercise real parsers, Pandas, FAISS serialization/search, HTTP handlers, and LangGraph routing. Most tests deliberately skip lifespan startup; a dedicated test exercises startup against a temporary index.

Socket connection/DNS/send attempts are blocked during application tests and checked even when application code swallows an exception. Provider credentials are blanked and `.env` loading is disabled. The runner performs no live model evaluation. The web-route fixture uses a reserved example URL and never fetches it.

An early test harness reset the entire environment after each test, removing native-library registration state and causing `OMP: Error #15`. Setting safety flags once for the test process fixed the observed abort. Tests bound OpenMP to one thread and reject `KMP_DUPLICATE_LIB_OK`; no unsafe bypass is enabled. Both Torch and FAISS ship OpenMP libraries in this installation, so passing these tests does not establish multithreaded native-runtime compatibility. This observation is a harness/environment limitation, not a newly established application vulnerability.

Installed packages also emit existing FastAPI lifespan, Starlette HTTPX, Google typing, and LangChain community deprecation warnings. These are retained in run logs and not suppressed. Dependency modernization belongs to a separate change.

## Evaluation data and optional real-model scoring

Labels live in `tests/fixtures/evaluation.json`; corpus files live in `tests/fixtures/corpus/`. The PDF case uses `tests.support.text_pdf()` to generate a real one-page PDF from the policy text. Every case identifies uploaded sources, history, expected source evidence, answer fragments, and a manual review rubric. The development set is intentionally small and visible; it must not be reused as an unbiased held-out benchmark.

Real-model execution is a separate opt-in activity and has not been performed. For each case, upload only its listed sources, start a fresh conversation, replay its history, ask the question, and capture actual answer/evidence. Record provider/model/revision, embedding configuration, corpus version, latency, and token/cost data where available alongside the predictions. Do not use golden labels to generate answers. This step provides an offline scorer, not an automatic live-provider adapter.

Normalize actual results into a JSON list, one entry per case:

```json
[
  {
    "id": "Q01",
    "answer": "The return deadline is 30 days from delivery.",
    "evidence_sources": ["policy.pdf"],
    "abstained": false
  }
]
```

This is a format example, not a measured prediction. If the current API lacks an abstention field, annotate it from the actual response and document that manual annotation. Score a complete prediction file explicitly:

```bash
python3 -m tools.evaluate_baseline --predictions /path/to/predictions.json --output .baseline-results/model-evaluation.json
```

The scorer rejects duplicate/unknown IDs, reports missing-case coverage, and counts omitted cases against aggregate scores. It measures source-level evidence recall and literal answer contracts, not semantic correctness. A matching number can still appear in an incorrect answer; apply every manual rubric. Unknown evidence sources are listed for review. Full coverage yields exit zero regardless of quality; quality gates require the metrics and manual review, not just the process exit code.

## Workload and acceptance targets

Measured baseline: 8 small physical corpus files, 9 logical source IDs including the generated PDF, 11 labeled questions; tests also generate long text and temporary indexes. Hardware recorded: macOS/Darwin arm64, 10 logical CPUs, CPython 3.14.6. Physical RAM capacity was unavailable in the sandbox; peak RSS, browser heap, and model latency were not measured.

Provisional development workload for later operational checks: one process, one concurrent chat initially, up to 100 documents / 10,000 chunks, then four concurrent clients and one ingestion job. Proposed targets: p95 lightweight state requests under 250 ms during ingestion; p95 warm document answer under 10 s on declared model/hardware; no cross-workspace access; all deterministic repair fixtures pass. These are **unmeasured planning targets**, not current performance or agreed production SLOs.

For model evaluation, require correct numeric results and evidence for all deterministic table fixtures, all gold sources for duplicate/conflict cases, abstention on the absent-answer case, and no hostile-instruction marker. Compare answer/evidence quality with the measured pre-change baseline before accepting retrieval changes. Larger held-out data and production workload requirements remain necessary before release.

## Next step

Step 0 is complete within the offline scope. Begin Step 1 only when requested: desired-behavior tests for invalid message schemas, blank queries, upload limits, request identity, and safe rendering, followed by the corresponding implementation. Do not turn unresolved diagnostic observations into passing regression assertions.
