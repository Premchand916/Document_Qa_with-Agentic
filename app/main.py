import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st

from ingestion.pdf_loader import load_pdf
from ingestion.semantic_chunker import semantic_chunk_documents
from langgraph_flow.graph_builder import build_graph
from vector_store.faiss_store import create_vector_store, load_vector_store


st.set_page_config(page_title="Document Intelligence Agent", layout="wide")

st.title("Document Intelligence Agent (LangGraph + RAG)")

# -----------------------------
# Initialize Session State
# -----------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    try:
        st.session_state.vectorstore = load_vector_store()
    except Exception as exc:
        st.session_state.vectorstore = None
        st.session_state["startup_error"] = str(exc)

if "last_result" not in st.session_state:
    st.session_state.last_result = {}

if "startup_error" in st.session_state:
    st.sidebar.error(st.session_state["startup_error"])


# -----------------------------
# Build LangGraph Workflow
# -----------------------------

graph = build_graph()


# -----------------------------
# Document Upload Section
# -----------------------------

st.sidebar.header("Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    all_docs = []

    for uploaded_file in uploaded_files:

        documents = load_pdf(uploaded_file)

        docs = semantic_chunk_documents(documents)

        all_docs.extend(docs)

    try:
        vectorstore = create_vector_store(all_docs)
        st.session_state.vectorstore = vectorstore
        st.sidebar.success("Documents embedded and saved to FAISS DB")

    except Exception as exc:
        st.sidebar.error(str(exc))
        st.stop()


# -----------------------------
# Chat Interface
# -----------------------------

st.subheader("Chat with your documents")

# Display previous chat messages
for msg in st.session_state.chat_history:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# Chat input (ONLY ONE INSTANCE)
query = st.chat_input(
    "Ask a question about your uploaded documents",
    key="document_chat_input"
)


if query:

    # Save user message
    st.session_state.chat_history.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.write(query)

    if st.session_state.vectorstore is None:

        response = "Please upload documents first."
        st.warning(response)

    else:

        state = {
            "query": query,
            "vectorstore": st.session_state.vectorstore,
            "chat_history": st.session_state.chat_history
        }

        with st.spinner("Agent is thinking..."):
            result = graph.invoke(state)

        response = result.get("response", "")
        st.session_state.last_result = result

    # Save assistant response
    st.session_state.chat_history.append(
        {"role": "assistant", "content": response}
    )

    with st.chat_message("assistant"):
        st.write(response)


result = st.session_state.last_result


# -----------------------------
# Agent Reasoning Trace
# -----------------------------

if "thoughts" in result:

    st.subheader("Agent Reasoning")

    for thought in result["thoughts"]:
        st.write(thought)


# -----------------------------
# Tool Execution Logs
# -----------------------------

if "tools_used" in result:

    st.subheader("Tool Execution")

    for tool in result["tools_used"]:
        st.write(tool)