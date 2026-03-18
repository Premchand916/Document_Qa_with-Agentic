from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def semantic_chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunked_docs = []

    for doc in documents:
        content_type = doc.metadata.get("content_type")
        if content_type in {"table_summary", "table_rows", "slide", "json_document"}:
            chunked_docs.append(doc)
            continue

        if len(doc.page_content) <= 900:
            chunked_docs.append(doc)
            continue

        chunks = splitter.split_text(doc.page_content)

        for chunk in chunks:
            chunked_docs.append(
                Document(
                    page_content=chunk,
                    metadata=doc.metadata
                )
            )

    return chunked_docs
