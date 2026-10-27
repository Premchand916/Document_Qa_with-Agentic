# Step 4 — Embedding compatibility results

Verified 27 October 2026. Step 4's **local compatibility gate passes**; shared deployment is not validated.

## Changes

- New format-2 corpus manifests store embedding dimension and fingerprint, exact `IndexFlatL2` metric, and chunker version. The fingerprint covers model name/options, loaded weight tensors, tokenizer vocabulary/special tokens, and SentenceTransformer module types. Synthetic embedding fixtures use class and public scalar settings.
- Publication checks the index dimension and metric, then reconstructs the candidate corpus before switching the active pointer. Snapshot and state readiness reject an incompatible encoder before native FAISS index loading or search.
- The embedding factory cache now keys model name, local-files setting, and optional `EMBEDDING_REVISION`. Changing `EMBEDDING_MODEL` in a running process selects the requested model instead of silently reusing the old instance.
- Offline loading now checks the project model cache when populated and otherwise uses the standard Hugging Face cache. This machine's standard cache supplied MiniLM without a download.
- The repository can reactivate a saved corpus version only after checksum, index, and current-encoder compatibility checks. A compare-and-swap pointer update rejects a concurrent change.

## Evidence and limits

The offline suite passes **105 tests**. The diagnostic harness reports **9 reproduced limitations**, **16 not reproduced**, one passing rejection control, and zero probe errors. E01, E02, and E03 no longer reproduce. Tests reject dimension and same-dimension coordinate changes before native index read; changed synthetic weights and tokenizer vocab produce different fingerprints. A synthetic rollback rejects the prior corpus with the wrong encoder, preserves the active version, then restores it with the matching encoder and searches successfully.

Two locally cached real models, MiniLM and BGE-small, each returned 384 dimensions. A MiniLM corpus reopened and searched with a fresh MiniLM instance. BGE-small was rejected against that saved corpus before native FAISS loading despite its matching dimension. MiniLM's first fingerprint took about 60 ms and a cached repeat about 3 ms in one local run; these are diagnostic timings, not workload targets.

Existing format-1 corpus manifests have no embedding identity and are rejected by format-2 code. **Re-upload original files** to rebuild them; original bytes were not retained by the application. Keep the old application and artifact backup until migration is rehearsed. No automatic conversion can prove an old index's encoder identity.

Multi-process deployment and a production-scale fingerprint cost have not been verified. Rollback is a repository operation, not an end-user API. Step 3 parser deadlines and JSON paths are in place; PDF provenance and resource measurement remain open. Step 5 deterministic table acceptance cases pass offline.
