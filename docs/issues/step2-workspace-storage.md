# Issue draft: isolate workspaces and publish corpus versions atomically

Local draft; not published. Step 2 replaces the global corpus/provider with a workspace repository. A trusted identity-to-workspace mapping authorizes every request. SQLite stores each workspace's provider and active corpus version; immutable artifact directories contain vectors, JSON documents, typed tables, and checksums.

Acceptance: separate workspaces cannot read or replace each other's corpus or preferences; tables and settings survive restart; a failed save preserves the previous version; conflicting publications cannot mix artifacts; an in-flight reader keeps its original version. Legacy pickle indexes are never automatically deserialized. Tests will inject write/commit failures and exercise concurrent publishers against a shared repository.

This change does not implement conversation or episodic memory, fix table-query semantics, or promise production multi-worker throughput. Authentication-proxy deployment remains a separate environment configuration.
