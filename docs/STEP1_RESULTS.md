# Step 1 — API Boundaries and Safe Rendering

Implemented and verified 26 October 2026. This is the historical Step 1 report; [Step 2](STEP2_RESULTS.md) now supersedes its shared-storage limitations and adds workspace membership configuration.

## Changes

- `app/schemas.py` validates nonblank queries, typed user/assistant history, known controls, strict booleans, and supported file-type hints. Unknown identity fields and system/developer/tool history roles are rejected.
- `app/security.py` establishes server-owned principal/workspace/request context, checks access before reading bodies, bounds incoming bytes before JSON/multipart parsing, and adds redacted errors with request IDs. Original request IDs supplied by clients are not trusted.
- Uploads are capped at **10 files**, **10 MiB per file**, and **25 MiB per entire multipart request**, including multipart overhead. JSON/other request bodies are capped at **128 KiB**. Queries and individual history messages allow **8,000 characters**; history allows **40 messages / 32,000 characters total**. These are resource controls, not tokenizer budgets.
- The upload handler reads files in bounded chunks, checks all sizes before document parsing, and closes temporary uploaded files. Invalid names, empty files, unsupported formats, and parser errors receive safe rejection responses. Raw exception details do not reach clients.
- QA serializes evidence as labeled JSON records and explicitly treats it as untrusted content. This is prompt hardening, not proof against all prompt injection. Provider configuration/memory exceptions now reach the API as redacted **503** errors; this also repairs P03 earlier than Step 10.
- UI history transmits role/content only and omits the current question already sent separately. Errors show safe messages and reference IDs; provider state changes only after a successful response. Browse/send/dismiss controls have accessible names and errors use an alert role.
- Markdown passes through DOMPurify with a narrow HTML/attribute allowlist; active images/frames/forms/scripts/framework directives are removed. Citation and answer URLs allow HTTP(S) only, without embedded credentials. External links use `noopener noreferrer`. Missing sanitizer support falls back to escaped text.
- JavaScript bundles are served locally: DOMPurify **3.4.17**, Marked **18.1.2**, Alpine **3.17.5**. npm tarball SHA-512 integrity was verified before extraction; bundle SHA-256 hashes, versions, source URLs, and licenses are in `app/static/vendor/`. Fonts still use the existing Google Fonts links.

## Local development

Access defaults to disabled. Enable development mode explicitly:

```bash
DOCQA_AUTH_MODE=local python3 -m uvicorn app.server:app --host 127.0.0.1 --port 8000 --no-proxy-headers
```

Alternatively set `DOCQA_AUTH_MODE=local` in the ignored `.env`. Both the socket peer and Host must be loopback/local, and browser Origin must match. Cross-site requests and forged identity headers are rejected. This mode represents one local developer, not separate OS users or browser accounts. Do not expose it through a reverse proxy.

## Shared access integration boundary

The chosen adapter delegates login/session validation to an established authentication proxy, such as OAuth2 Proxy plus its reverse proxy. No homegrown password store or JWT parser was introduced. OAuth2 Proxy can provide an authenticated user header; the surrounding trusted proxy must overwrite inbound identity headers and add a private server-to-server secret. [OAuth2 Proxy configuration](https://oauth2-proxy.github.io/oauth2-proxy/configuration/overview/).

Required server configuration:

- `DOCQA_AUTH_MODE=auth_proxy`.
- `DOCQA_PUBLIC_ORIGIN`: exact HTTPS origin, without path or trailing slash.
- `DOCQA_PROXY_SECRET`: a securely generated secret of at least 32 characters, kept outside Git and browsers.
- `DOCQA_PROXY_PEERS`: comma-separated actual socket-peer IPs; defaults to loopback.
- `DOCQA_ALLOWED_SUBJECTS`: comma-separated authenticated subject IDs from that configured identity source. Prefer stable IDs rather than display names.
- `DOCQA_WORKSPACE_ID`: the single server-owned workspace identifier.

The trusted proxy must forward the public Host, set `X-Auth-Request-User` from verified identity, and inject `X-DocQA-Proxy-Secret` from its server configuration. Strip client-provided copies; never expose the secret in a response. Run Uvicorn with `--no-proxy-headers` so peer checks use the socket peer. Keep the backend unreachable except from that proxy; use a protected transport if it is not on the same machine. Invalid peer/host, secret, subject, or Origin is rejected before body processing. Incomplete configuration fails closed.

All allowlisted subjects intentionally share the **one existing corpus and provider preference**. Per-user/per-workspace storage, database membership, revocation storage, and provider isolation remain Step 2 work. The adapter contract was tested with synthetic headers; no real identity provider, proxy deployment, login, or account permissions were configured in this session.

## Verification

Three desired-behavior tests first failed against the original implementation: blank query accepted, forged/null history role accepted, and excessive file count accepted. The repairs were then checked with:

```bash
python3 -m tools.run_baseline --output-dir .baseline-results/step1
python3 -m compileall -q agents app embeddings ingestion langgraph_flow retrieval tools utils vector_Store memory tests
node --check app/static/safe-render.js
git diff --check
```

**50 tests pass**, including the original 28 controls and 22 new boundary tests. No skips or expected failures. Chunked requests cannot bypass the byte limit, file-count/size rejection precedes document parsing, and errors do not echo synthetic secrets. Diagnostics now show **22 reproduced, 3 no longer reproduced, 1 control passed, 0 errors**. M02 and V01 are fixed by validation; P03 is fixed by provider-error handling. Retained results: [summary](step1/summary.json), [diagnostics](step1/diagnostics.json). The original baseline evidence is preserved.

Chrome executed **14/14 sanitizer checks** using the real local bundles: event handlers, unsafe/encoded/data URLs, SVG, frames, scripts, framework directives, DOM clobbering, safe formatting/links, and missing-sanitizer fallback. Desktop UI smoke checks covered synthetic upload, a suggested-question answer, visible source evidence, and an oversized-query validation error with a reference ID. The generated answer contained hostile image markup that was removed during rendering.

Reproduce browser checks without loading real models or user indexes:

```bash
python3 -m tools.preview_step1 --port 8765
```

Open `http://127.0.0.1:8765/_checks`, then `/`. Upload `tests/fixtures/corpus/policy.txt`. This preview uses temporary indexes and stubbed model/search calls; its check route is not part of the production app.

## Remaining limits and follow-ups

- At the **393 × 852 iPhone 16 viewport**, the existing fixed three-column layout squeezed the chat panel out of view. Desktop security changes work; responsive layout needs a separate fix. No responsive redesign was included here.
- Upload limits do not yet bound decompressed Office archives, parser/OCR time, simultaneous jobs, or total process RAM. Those belong to Steps 3 and 10.
- Same-origin XSS defenses now have controlled fixture evidence, not a guarantee against every payload or future dependency flaw. The CSP currently restricts framing, objects, and base URLs; a strict script policy requires a later Alpine/inline-script migration.
- Conversation memory, table correctness, atomic persistence, embedding compatibility, reranker recovery, and the remaining audit findings are unchanged. Live answer quality and real identity-provider integration remain untested.
- The legacy Streamlit application does not use this boundary. Use the FastAPI entry point for these controls.

Next implementation step: workspace repositories and atomic corpus publication (Step 2).
