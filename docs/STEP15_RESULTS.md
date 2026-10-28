# Step 15 results — trusted review authorization and ranking evidence

Verified 28 October 2026. Ollama fine-tuning remains paused.

## Reviewer authorization

### Boundary

Reviewer authority comes only from server configuration:

```dotenv
DOCQA_WORKSPACE_MEMBERSHIPS={"finance":["alice","reviewer@example.com"]}
DOCQA_WORKSPACE_REVIEWERS={"finance":["reviewer@example.com"]}
```

A reviewer must also be a member of the same workspace. Unknown workspaces, empty reviewer lists, non-member reviewers, client identity fields, and forged proxy identities fail closed. Local mode grants no reviewer role unless the server-owned settings explicitly do so.

### Workflow

1. User submits correction; episode remains `unverified`.
2. Authorized reviewer loads `GET /api/review/episodes` for active workspace.
3. Reviewer approves or rejects through `POST /api/review/episodes/{episode_id}`.
4. Server finds episode owner internally; client cannot choose or forge owner.
5. Status update and audit row commit in one transaction.
6. Concurrent/stale second review fails because endpoint accepts only an `unverified` episode.

Studio now shows Feedback Review only when authenticated context contains server-derived `can_review=true`.

## Label-blind semantic ranking evaluation

- Inference tasks and expected labels live in separate JSON bundles.
- Task loader rejects label-like fields such as `expected_key`, `answer`, and `target`.
- Inference produces ranked memory keys before scorer reads labels.
- Cases cover older relevant keys, relevance found in values, and no-term recency fallback.
- Development Recall@1: 1.0 against 1.0 target.

This proves deterministic ranker behavior on frozen synthetic cases. It is development evidence, not real-user or model-quality evidence.

## Secret-pattern hardening

Semantic memory now also rejects common secret assignments, credential-bearing URLs, and JWT-like values in addition to private keys and provider-token patterns. This reduces accidental persistence; it does not replace enterprise DLP or secret scanning.

## Verification

- Full baseline: 194 tests; all seven stages passed.
- Diagnostics: zero reproduced defects, 25 not reproduced, one passing rejection control, zero failed controls, zero probe errors.
- Reviewer tests prove role membership, fail-closed invalid configuration, ordinary-member denial, owner-forgery rejection, queue processing, audit identity, and queue removal.
- Memory retrieval evaluator passes Recall@1 1.0 and rejects label leakage.
- Compilation and whitespace checks pass.

## Remaining boundaries

1. Authentication proxy/identity-provider deployment remains external; repository now consumes its trusted subject and configured reviewer role.
2. Procedural-memory approval needs a separate administrator role, signed procedure artifacts, and rollback. Reviewer role does not grant it.
3. Multi-host storage still requires a PostgreSQL migration contract, locking/load tests, and rollback rehearsal.
4. Production must wire and monitor the existing cleanup command.
5. Lexical ranking still lacks general synonym understanding; do not add embeddings until a larger label-blind set proves benefit.
6. Pattern-based secret rejection is broader but remains incomplete compared with full DLP.
7. Ollama answer-usefulness evaluation and fine-tuning remain separate future phases.
