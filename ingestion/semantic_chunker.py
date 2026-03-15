from langchain_core.documents import Document


def _find_split_end(text, start, chunk_size):
    end = min(start + chunk_size, len(text))

    if end >= len(text):
        return len(text)

    window = text[start:end]

    for separator in ("\n\n", "\n", ". ", " "):
        split_at = window.rfind(separator)
        if split_at >= chunk_size // 2:
            return start + split_at + len(separator)

    return end


def _split_text(text, chunk_size=800, chunk_overlap=100):
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        end = _find_split_end(cleaned_text, start, chunk_size)
        chunk = cleaned_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def semantic_chunk_documents(text, metadata):
    chunks = _split_text(
        text,
        chunk_size=800,
        chunk_overlap=100
    )

    documents = []

    for chunk in chunks:
        documents.append(
            Document(
                page_content=chunk,
                metadata=metadata
            )
        )

    return documents
