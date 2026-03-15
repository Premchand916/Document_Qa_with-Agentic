def retriever_agent(state):

    query = state["query"]
    vectorstore = state["vectorstore"]

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    docs = retriever.invoke(query)

    state["documents"] = docs

    return state