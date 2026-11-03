# ChatGPT-style system evidence ledger

Source lane: `SOURCE`. Repository target: commit `31e0c3a69b8e28d02530159f1bc77d3ca1226b31`, plus current uncommitted architecture documentation. Reviewed 3 November 2026. REA MCP is connected, but its artifact/runtime tools were intentionally not invoked because the authorized source repository answers the current architecture question.

Classes: **Observed** means direct source/command/document evidence; **Derived** means deterministic consequence; **Inferred** needs implementation or runtime proof; **Unknown** is not established.

| ID | Claim | Class | Evidence | Coverage and limitation | Falsifying check |
|---|---|---|---|---|---|
| E-001 | FastAPI is the active API composition root and SPA host. | Observed | `app/server.py:78-140`, `app/server.py:703-735`, `app/server.py:861-898` | Source path only; deployment topology not observed. | Inspect deployed process/routes. |
| E-002 | Chat supports durable owner-scoped turns, leases, idempotency, summaries, retention, and restart recovery. | Observed | `storage/conversations.py:36-108`, `storage/conversations.py:171-294`; conversation tests | SQLite single-host behavior; multi-host database not proven. | Multi-process and PostgreSQL integration tests. |
| E-003 | Active LangGraph is a fixed acyclic document/table/web workflow, not a general iterative agent runtime. | Observed + derived | `langgraph_flow/graph_builder.py:13-77` | Static source proves configured graph, not production invocation counts. | Runtime trace on deployed build. |
| E-004 | General chat bypasses retrieval when no grounded source exists; grounded graph handles document/table/web paths. | Observed | `app/server.py:515-641` | Model quality/provider behavior not established by source. | API test with real configured model. |
| E-005 | Current prompt skill loader selects one operator/repository file and has no registry, dependency, permission, or activation lifecycle. | Observed + derived | `utils/prompt_skill.py:13-61`, repository inventory | Absence claim scoped to inspected repo at target commit. | Locate another registered skill subsystem or runtime configuration. |
| E-006 | No repository-local `SKILL.md`, plugin manifest, or MCP configuration exists. | Observed | `find` inventory returned no matching files outside ignored/tool directories. | External deployment mounts remain unknown. | Inspect deployment capability directories. |
| E-007 | Authentication is local or trusted-proxy based with server-derived workspace membership and reviewer authority. | Observed | `app/security.py:26-122` | Real identity-provider/proxy deployment not observed. | Staging OIDC/proxy test. |
| E-008 | UI is chat-first with separate knowledge management, but remains a monolithic Alpine HTML file. | Observed | `app/static/index.html:455-660`, `app/static/index.html:790-860`; 1,221-line count | Browser/device behavior beyond prior local checks remains unverified. | Automated browser accessibility/responsive suite. |
| E-009 | No voice/Wispr/audio implementation exists in this repository. | Observed | Case-insensitive repo search for `wispr`, `voice`, `speech`, `audio`, `transcrib`, `whisper`, `realtime`, `webrtc`, `tts`, `stt` returned only roadmap text and unrelated identifiers. | External voice repository is outside supplied scope. | Inspect separate voice repository when access is granted. |
| E-010 | Public Wispr material establishes dictation into applications, not a reviewed public embeddable Nexus API. | Observed + unknown | `https://wisprflow.ai/developers`; `https://docs.wisprflow.ai/articles/3152211871` | A private/partner API may exist; public review does not prove absence. | Obtain vendor API/enterprise documentation. |
| E-011 | Native low-latency voice needs a separate realtime session/transport layer while business tools remain server-controlled. | Observed public contract + derived | `https://developers.openai.com/api/docs/guides/realtime`; server-controls documentation | OpenAI contract is design reference, not required provider choice. | Prototype against selected voice provider. |
| E-012 | Current deterministic baseline passes 200 tests and seven stages. | Observed execution | `python3 -m tools.run_baseline`, executed during this architecture review | Offline fixtures block real providers/network; no production load. | Repeat after every implementation slice and run opt-in provider E2E. |

## Derived architecture decisions

- **D-001:** retain ingestion, retrieval, evidence, conversations, and memory; add a capability/event kernel around them.
- **D-002:** implement skills, connectors, streaming, tools, plugins, artifacts, coding, then subagents in dependency order.
- **D-003:** treat Wispr Flow as external dictation unless a supported vendor API is provided.
- **D-004:** keep native voice outside core Steps 20–31, but require it to reuse thread, policy, event, tool, approval, and audit contracts later.
- **D-005:** keep modular monolith until measured multi-host or scaling requirements justify service extraction.

## Unknowns requiring later evidence

- Production identity provider, deployment topology, worker count, traffic, latency SLO, and recovery objectives.
- First enterprise content provider: SharePoint, Google Drive, OneDrive, S3, or another governed source.
- Secret-vault and PostgreSQL deployment choices.
- Local Ollama model's reliable structured tool-calling and streaming behavior on final hardware.
- Separate voice repository architecture, provider, latency, interruption, telephony, retention, and consent requirements.
- Whether Wispr offers a supported enterprise/developer API suitable for direct product integration.
