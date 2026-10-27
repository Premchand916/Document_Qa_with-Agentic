"""Bound every embedded chunk, including tables, slides, and JSON."""

import hashlib
import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CONTENT_TOKEN_BUDGET = 240  # Reserve 16 of MiniLM's 256 tokens for special tokens.
CHUNKER_VERSION = "token-budget-v1"


def semantic_chunk_documents(documents, tokenizer=None):
    """Use the encoder tokenizer when available; otherwise bound UTF-8 bytes."""
    def length(value):
        if tokenizer is None:
            return len(value.encode("utf-8"))
        return len(tokenizer.encode(value, add_special_tokens=False))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONTENT_TOKEN_BUDGET,
        chunk_overlap=32,
        length_function=length,
        separators=["\n\n", "\n", " ", ""],
    )
    chunked_docs = []
    occurrences = {}
    for document in documents:
        pieces = [document.page_content] if length(document.page_content) <= CONTENT_TOKEN_BUDGET else splitter.split_text(document.page_content)
        for index, piece in enumerate(pieces):
            if length(piece) > CONTENT_TOKEN_BUDGET:
                raise ValueError("Document chunk exceeds tokenizer budget")
            metadata = dict(document.metadata)
            identity = {key: metadata.get(key) for key in ("source_id", "source", "page", "sheet", "row_start", "row_end", "json_pointer", "line_start", "line_end")}
            identity.update(chunk_index=index, content_hash=hashlib.sha256(piece.encode("utf-8")).hexdigest())
            base_id = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()
            occurrence = occurrences.get(base_id, 0)
            occurrences[base_id] = occurrence + 1
            metadata["chunk_id"] = f"{base_id}:{occurrence}"
            metadata["chunk_index"] = index
            chunked_docs.append(Document(page_content=piece, metadata=metadata))
    return chunked_docs
