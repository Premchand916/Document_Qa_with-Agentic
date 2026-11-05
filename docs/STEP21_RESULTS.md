# Step 21 Results — Versioned Skill Runtime

**Status:** implemented and verified locally  
**Date:** 3 November 2026

## Built

- Strict `SkillManifest`, descriptor, loaded-package, and activation contracts.
- Repository-local `.agents/skills/*/SKILL.md` discovery with maximum skill/file/resource/package limits.
- Restricted front-matter parser; strict required fields; semantic versions; validated identifiers, triggers, capabilities, and resources.
- SHA-256 package locks covering `SKILL.md` and declared resources.
- Traversal, symlink, undeclared-file, invalid UTF-8, digest mismatch, instruction-override, oversized-package, and duplicate/invalid package rejection.
- Progressive loading: public discovery returns metadata/digest only; instructions and resources load only during explicit activation and are revalidated.
- Workspace-scoped enablement and audit storage. Existing approval fails closed after version/digest change.
- Capability-policy intersection: skill dependencies must already exist and be authorized; instructions cannot create permissions.
- Deterministic matcher with ambiguity rejection for future routing. Production chat uses explicit `skill_id` only.
- `GET /api/skills` read-only inventory plus prompt-library skill metadata.
- Built-in locked `prompt-library` package. UI library selection sends its skill identity; manual typing clears selection.
- Added locked `build-application-safely` package for risk mapping, vertical slices, root-cause repair, layered verification, truthful evidence boundaries, and durable handoff.
- Added canonical development checklist covering authority, desired behavior, breakpoints, implementation, verification, failure repair, completion, and handoff.
- Declared resources load only after activation and share a strict 24 KB active-guidance prompt budget.
- Skill frontmatter now follows standard `name`/`description` fields with runtime-specific values inside strict JSON `metadata`; both Codex and application validators accept it.
- Safe activation guidance enters prompts below system/security/evidence rules. Activation emits `capability.selected` with identity/version/digest only.

## Validation

- Step 21 adversarial suite: 7/7 tests pass.
- Impacted API, graph, conversation, procedural-memory, event, and capability suites: 68/68 tests pass.
- Full baseline: 219 tests and all seven stages pass.
- Compile and whitespace checks pass.
- Local FastAPI smoke test returned 200 for `/api/skills` and `/api/prompt-library`, with locked `prompt-library` enabled and dependency-authorized.
- Both repository skills pass standard Codex `quick_validate` and application digest/runtime validation.

## Resolved limitations

- Static prompt suggestions now have a versioned skill identity and activation audit.
- Silent package modification is detected by digest locks; workspace approvals invalidate on changed content/version.
- Cross-workspace enablement leakage, unrestricted context loading, path escape, symlinks, undeclared resources, ambiguous routing, and capability escalation fail closed.
- Disabled or dependency-unauthorized skills never activate or enter model context.

## Deliberate limitations

- No remote skill installation, marketplace, arbitrary scripts, or hot-loaded external roots.
- No API mutation for enablement. State changes remain an internal/operator boundary until production roles and approval workflow exist.
- Skill resources are UTF-8 guidance only; they are not executable code or factual evidence.
- Automatic matcher exists but production activation remains explicit to avoid surprising routing.
- External connectors, SSE/Stop, tool loop, plugins/MCP, coding tools, and subagents remain later steps.
- Browser automation host was unavailable for a live visual smoke test. Static UI assertions and HTTP integration tests pass; no layout structure changed.

## Next gate

Step 22: connector registry and one read-only SOP/SOW source. Add owner-scoped grants, sources, immutable snapshots, sync runs/cursors, ACL/freshness/revocation/deletion propagation, and citation provenance. Connector content remains untrusted evidence; credentials never enter skills, prompts, events, or SQLite plaintext.
