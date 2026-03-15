def retrieve_documents(query, vectorstore, k=4):

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    docs = retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content if hasattr(doc, "page_content") else str(doc)
        for doc in docs
    )

    return context