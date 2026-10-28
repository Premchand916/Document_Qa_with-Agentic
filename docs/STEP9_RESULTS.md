# Step 9 results — selective episodic feedback

Verified 28 October 2026. Local deterministic gate passes. Episodic recall is opt-in and disabled for unverified feedback.

## Implemented

- Users explicitly submit a correction from a completed turn. Ordinary chat is never copied into episodic memory automatically.
- Server derives task key, original result, tool context, sources, citations, and corpus version from the authorized completed turn. Public request schema accepts only conversation ID, request key, and correction; extra trust/result fields are rejected.
- New feedback is always `unverified`. Only internal repository validation can mark it `verified` or `rejected`; no public verification route exists without an administrator/validator role.
- Recall requires exact workspace, principal, normalized task key, current corpus version, verified status, and unexpired timestamp. Default expiry is 180 days through `DOCQA_EPISODE_RETENTION_DAYS`.
- Recalled corrections enter prompt guidance with an explicit “not evidence” label. They are not documents, citations, or retrieval evidence.
- Episodes share the conversation SQLite database and reference their conversation with `ON DELETE CASCADE`. Conversation deletion/retention removes linked episodes. Episode deletion leaves transcript data intact.
- Owner-scoped APIs support create, list, and delete. Duplicate feedback for one completed turn returns 409.
- UI offers “Submit correction for review” and clearly reports that saved feedback remains unverified.

## Verification

- Full offline suite: **148 tests pass**.
- Verified same-task/same-corpus correction is recalled and appears in prompt guidance.
- Unverified, rejected, expired, wrong-task, wrong-corpus, cross-principal, and cross-workspace episodes do not recall.
- Public verification forgery returns 422. Duplicate feedback returns 409.
- Conversation deletion cascades episodes; deleting an episode preserves the completed transcript.
- Live synthetic browser check submitted a correction from Studio, received HTTP 200 from `POST /api/episodes`, and displayed the unverified-feedback status.
- Historical diagnostics remain zero reproduced defects, 25 not reproduced, one passing rejection control, and zero probe errors.

## Remaining limits

- Verification is intentionally internal until trusted reviewer/admin roles exist. Production needs a review queue, audit identity, reason, and timestamp before exposing approval controls.
- Exact normalized task matching is conservative. Add an episode embedding index only after labeled evaluation shows benefit without cross-task leakage.
- Same-corpus recall excludes useful corrections after re-indexing. Future migration may revalidate episodes against stable source hashes and evidence IDs.
- Expired episode purge runs at startup. Step 10 should schedule continuous cleanup and metrics.
- Corrections guide generation but cannot repair missing/bad evidence. Fine-tuning remains gated on evaluated, consented, redacted, verified outcomes.

Next: Step 10 durable jobs, cancellation boundaries, streaming status, per-workspace fairness, stage metrics, and multi-process load tests.
