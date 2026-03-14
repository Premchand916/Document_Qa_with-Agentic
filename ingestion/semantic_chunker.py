from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def semantic_chunk_documents(text, metadata):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    documents = []

    for chunk in chunks:
        documents.append(
            Document(
                page_content=chunk,
                metadata=metadata
            )
        )

    return documents