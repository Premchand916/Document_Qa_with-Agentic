"""Stable evidence identity, conflict signals, and citation validation."""

import hashlib
import json
import re


_CITATION = re.compile(r"\[(E-[a-f0-9]{12})\]")
_VALUE = re.compile(r"\b\d+(?:\.\d+)?(?:\s*(?:%|days?|weeks?|months?|years?))?\b", re.IGNORECASE)


def _evidence_id(document) -> str:
    metadata = document.metadata
    stable = metadata.get("chunk_id") or json.dumps({
        "source_id": metadata.get("source_id"),
        "source": metadata.get("source"),
        "page": metadata.get("page"),
        "sheet": metadata.get("sheet"),
        "rows": metadata.get("rows"),
        "url": metadata.get("url"),
        "content": document.page_content,
    }, sort_keys=True, default=str)
    return "E-" + hashlib.sha256(str(stable).encode("utf-8")).hexdigest()[:12]


def prepare_evidence(state: dict) -> list:
    """Attach repeatable IDs and request-owned origin/version fields."""
    documents = list(state.get("documents") or [])
    version = state.get("corpus_version")
    for document in documents:
        metadata = document.metadata
        metadata["evidence_id"] = metadata.get("evidence_id") or _evidence_id(document)
        metadata["origin"] = "web" if metadata.get("content_type") == "web_result" else "local"
        if metadata["origin"] == "local" and version:
            metadata["corpus_version"] = version
    state["documents"] = documents
    return documents


def detect_evidence_conflict(documents: list) -> dict:
    """Flag differing explicit values across sources; do not guess resolution."""
    by_source = {}
    for document in documents:
        source = str(document.metadata.get("source") or "Unknown")
        values = {match.group(0).casefold() for match in _VALUE.finditer(document.page_content)}
        if values:
            by_source.setdefault(source, set()).update(values)
    distinct = {tuple(sorted(values)) for values in by_source.values()}
    if len(by_source) > 1 and len(distinct) > 1:
        return {"status": "possible_conflict", "sources": sorted(by_source),
                "values_by_source": {source: sorted(values) for source, values in sorted(by_source.items())}}
    return {"status": "consistent_or_not_comparable", "sources": sorted(by_source)}


def finalize_answer(state: dict) -> dict:
    """Expose only citations that resolve to current response evidence."""
    documents = prepare_evidence(state)
    evidence = {doc.metadata["evidence_id"]: doc for doc in documents}
    response = str(state.get("response") or "").strip()
    cited = list(dict.fromkeys(_CITATION.findall(response)))
    bracketed_evidence = set(re.findall(r"\[(E-[^\]]+)\]", response))
    invalid = sorted(bracketed_evidence - set(evidence))
    if invalid:
        state["response"] = "Answer could not be verified against the current evidence. Please retry."
        state["result_status"] = "citation_error"
        cited = []
    elif response and documents and not cited:
        cited = [documents[0].metadata["evidence_id"]]
        state["response"] = f"{response}\n\nEvidence reviewed: [{cited[0]}]"
    state["citations"] = [{"evidence_id": item, "source": evidence[item].metadata.get("source", "Unknown"),
                           "location": evidence[item].metadata.get("page", evidence[item].metadata.get("sheet")),
                           "origin": evidence[item].metadata.get("origin"),
                           "corpus_version": evidence[item].metadata.get("corpus_version")}
                          for item in cited if item in evidence]
    state["evidence_status"] = detect_evidence_conflict(documents)
    if state.get("result_status") is None:
        state["result_status"] = "answered" if response and documents else "insufficient_evidence"
    return state
