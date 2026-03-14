def search_documents(vectorstore, query):

    docs = vectorstore.similarity_search(query, k=5)

    context = "\n".join([doc.page_content for doc in docs])

    return context