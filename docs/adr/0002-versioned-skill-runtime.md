# ADR 0002: Skills are locked guidance, never executors

**Status:** accepted locally  
**Date:** 3 November 2026

## Context

Prompt templates existed as UI text, while optional procedural prompt loading had no shared manifest, workspace state, dependency policy, or activation audit. Adding arbitrary skill scripts would bypass the Step 20 capability boundary.

## Decision

- A skill is a repository-local `SKILL.md` package containing bounded guidance and declared resources.
- Every package uses strict metadata, semantic version, declared capability dependencies, and a SHA-256 lock.
- Discovery exposes metadata only. Instructions/resources load only after explicit selection and revalidation.
- Package directories, manifests, locks, and resources must be regular non-symlink files. Traversal, undeclared files, duplicate/invalid identities, oversized content, and instruction-override patterns fail closed.
- Workspace state may enable or disable a reviewed digest. A version/digest change invalidates existing approval.
- Skill instructions never grant authority. Activation intersects dependencies with server capability policy.
- Production chat activates only an explicitly supplied `skill_id`; automatic matcher output is not trusted for execution.
- Activation writes a safe `capability.selected` turn event containing identity/version/digest, never instructions.

## Consequences

Prompt-library workflows become auditable without enabling code execution. Skill changes require a reviewed lock update and workspace reapproval when an explicit state exists. This step does not install remote skills, execute scripts, grant tools, or provide plugin/MCP behavior.
