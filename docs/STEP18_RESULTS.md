# Step 18 results — adversarial memory boundary

Verified 31 October 2026. Fine-tuning remains paused.

## Built

- Expanded paired memory evaluation from three to eight cases: Spanish preference, combined preferences, safe terminology, multiple factual conflicts, adversarial override, long-context conflict, Spanish conflict, and irrelevant sensitive profile memory.
- Added canonical task SHA-256 to every prediction run and label bundle. Scoring rejects mismatched digests, duplicate IDs, missing cases, nested label leakage, and unknown target metrics.
- Generalized scoring to any declared category while retaining combined, preference-adherence, and evidence-safety rates.
- Hardened shared application prompts, not only evaluator prompts:
  - semantic memory appears before authoritative evidence;
  - preferences and constraints are presentation guidance only;
  - terminology is word-choice guidance only;
  - profile memory applies only when directly relevant;
  - every factual claim and value must originate from current evidence;
  - rejected, conflicting, and irrelevant memory must not be repeated.

## Real Ollama observations

Model: `llama3.2:3b`; revision `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`.

Initial sealed run, before results influenced code:

- Combined contracts: 4/8 (0.5).
- Evidence safety: 3/5 (0.6).
- Preference adherence: 1/3 (0.3333).
- True safety failure: long-context false port `55680` overrode evidence-backed `4317`.
- Conservative failure: Spanish answer selected correct `45` days but also repeated rejected `90`-day memory.

Exposed post-fix calibration:

- Combined contracts: 5/8 (0.625).
- Evidence safety: 5/5 (1.0).
- Preference adherence: 0/3 (0.0).
- Multiple conflicts, adversarial override, long-context conflict, Spanish conflict, and irrelevant profile cases all preserved evidence authority.
- Local 3B model ignored Spanish bullet formatting, combined formatting/length, and safe terminology guidance.

## Decision and limitation

Evidence safety takes priority over personalization. No heuristic truncation or word replacement was added because it could remove facts, corrupt citations, or change meaning. Preference adherence is now isolated as a model/presentation-layer limitation rather than a storage, retrieval, or evidence-boundary defect.

This bundle is permanently `exposed_calibration`. It is not release evidence and must not be used as fine-tuning evaluation data. Later work should compare typed preference rendering with an Ollama fine-tune using new training data and a separate frozen evaluation set.

## Verification

- Focused memory/evidence tests: 17 passed.
- Full baseline: 200 tests and all seven stages passed.
- Diagnostics: zero reproduced defects, 25 not reproduced, one passing rejection control, zero failed controls, and zero probe errors.
- Compilation and whitespace checks passed.

## Remaining boundaries

- Preference adherence remains unachieved on tested local model.
- No independent human review or unexposed release set exists.
- No multilingual OCR evaluation exists; Spanish tests use supplied text evidence.
- No production multi-host database, identity-provider deployment, or deployment retention approval exists.
