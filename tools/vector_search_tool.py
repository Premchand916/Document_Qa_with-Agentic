def search_documents(vectorstore, query, metadata_filter=None):

    docs = vectorstore.similarity_search(
        query,
        k=5,
        filter=metadata_filter
    )

    context = "\n".join([doc.page_content for doc in docs])

    return context