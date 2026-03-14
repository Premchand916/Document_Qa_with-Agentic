from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def semantic_chunk_documents(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_text(text)

    documents = [Document(page_content=chunk) for chunk in chunks]

    return documents