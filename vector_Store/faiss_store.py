from langchain.vectorstores import FAISS
from embeddings.google_embedding import get_embedding_model

embedding_model = get_embedding_model()

def create_vector_store(documents):

    vectorstore = FAISS.from_documents(
        documents,
        embedding_model
    )

    return vectorstore