# Step 17 results — paired Ollama memory-quality evaluation

Verified 31 October 2026. Ollama fine-tuning remains paused.

## What was built

- A paired runner invokes each task with semantic memory off and on through the existing Ollama orchestrator.
- Inference reads only task inputs. Required/forbidden fragments, regexes, categories, and targets live in a separate label file loaded after predictions exist.
- Each prediction records the provider, model, exact local model digest, latency, and captured input/output token use.
- Scoring separates preference adherence from factual evidence safety while retaining a combined contract gate.
- Tests reject label fields in the task bundle and prove separate label-blind inference and scoring.

## Observed result

Model: `llama3.2:3b`, revision `a80c4f17acd5...b72`.

- Preference: memory-off returned `30 days.`; memory-on returned `• 30 days`.
- Conflict: the answer retained the evidence-backed 30-day deadline and did not use the conflicting remembered 60-day value.
- Irrelevant memory: the answer returned `help@example.test` and did not expose the unrelated favorite color.
- Combined contract: 3/3. Preference adherence: 1.0. Evidence safety: 1.0.

The first scorer accepted only ASCII `-` and `*` bullets, so it incorrectly rejected the valid Unicode bullet. The corrected label accepts the common `•` form. This was an evaluator-boundary correction; the recorded model answer was not changed.

## Evidence boundary

This dataset is development calibration, not release evidence. Its result caused one prompt clarification and one scoring correction. It must not be used to claim held-out quality or as a fine-tuning training set. A fresh frozen candidate set must be evaluated once after its tasks and labels are sealed.

## Verification

- Focused unit tests: 2 passed.
- Full baseline: 198 tests and all seven stages passed.
- Diagnostics: zero reproduced defects, 25 not reproduced, one passing rejection control, zero failed controls, and zero probe errors.
- Whitespace and compilation checks passed.

## Remaining boundaries

- Three cases are enough to validate the harness, not to estimate production quality.
- No multilingual, long-context, multiple-conflict, preference-combination, or adversarial-memory coverage exists yet.
- No independent human review or release-candidate dataset exists.
- Fine-tuning remains paused until a larger evaluation attributes a measured gap to model behavior rather than prompt, retrieval, evidence, or scoring defects.
