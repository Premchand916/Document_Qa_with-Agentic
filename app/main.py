import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st

from ingestion.pdf_loader import load_pdf
from ingestion.semantic_chunker import semantic_chunk_documents
from langgraph_flow.graph_builder import build_graph
from vector_store.faiss_store import create_vector_store, load_vector_store


st.set_page_config(page_title="Document Intelligence Agent", layout="wide")

st.title("Document Intelligence Agent (LangGraph + RAG)")

# Build LangGraph workflow
graph = build_graph()

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

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

    for file in uploaded_files:

        with st.spinner(f"Processing {file.name}..."):

            text = load_pdf(file)

            metadata = {
                "document_name": file.name,
                "document_type": "pdf"
            }

            docs = semantic_chunk_documents(text, metadata)

            all_docs.extend(docs)

    try:
        vectorstore = create_vector_store(all_docs)
    except Exception as exc:
        st.sidebar.error(str(exc))
        st.stop()

    st.session_state.vectorstore = vectorstore

    st.sidebar.success("Documents embedded and saved to FAISS DB")


# -----------------------------
# Chat Interface
# -----------------------------

st.subheader("Chat with your documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_input = st.chat_input("Ask a question about your uploaded documents")

if user_input:

    if st.session_state.vectorstore is None:
        st.warning("Please upload documents first.")
        st.stop()

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # Prepare state for agent
    state = {
        "query": user_input,
        "vectorstore": st.session_state.vectorstore
    }

    with st.spinner("Agent is thinking..."):

        result = graph.invoke(state)

        response = result.get("response", "No response generated.")
        st.session_state.last_result = result

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    with st.chat_message("assistant"):
        st.write(response)


result = st.session_state.last_result

# -----------------------------
# Reasoning Trace
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
