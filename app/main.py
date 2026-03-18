import os
import sys
import traceback
import html
import re

# Fix import paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st

from agents.question_generator import generate_questions
from ingestion.pdf_loader import load_pdf
from ingestion.semantic_chunker import semantic_chunk_documents
from langgraph_flow.graph_builder import build_graph
from vector_store.faiss_store import create_vector_store, load_vector_store

st.set_page_config(page_title="Document Intelligence Workspace", layout="wide")


def record_error(stage, exc):
    st.session_state.last_error = {
        "stage": stage,
        "message": str(exc),
        "details": traceback.format_exc(),
    }


def clear_error():
    st.session_state.last_error = None


def uploaded_file_signature(files):
    return tuple((file.name, getattr(file, "size", None)) for file in files)


def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def summarize_words(text, max_words=20):
    words = clean_text(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(".,;:") + "..."


def build_source_cards(documents):
    cards = []
    seen = set()

    for index, document in enumerate(documents or []):
        metadata = getattr(document, "metadata", {}) or {}
        source_name = metadata.get("source", "Unknown document")
        page = metadata.get("page", "N/A")
        excerpt = clean_text(getattr(document, "page_content", ""))

        if not excerpt:
            continue

        dedupe_key = (source_name, page, excerpt[:160])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        title = source_name if page in {None, "", "N/A"} else f"{source_name} - Page {page}"
        cards.append(
            {
                "id": f"{source_name}-{page}-{index}",
                "title": title,
                "source": source_name,
                "page": page,
                "summary": summarize_words(excerpt, max_words=20),
                "excerpt": excerpt,
            }
        )

    return cards


SOURCE_STOPWORDS = {
    "about", "after", "again", "being", "below", "between", "could", "from",
    "have", "into", "just", "more", "most", "other", "same", "some", "such",
    "than", "that", "them", "then", "there", "these", "they", "this", "very",
    "what", "when", "where", "which", "while", "with", "would", "your"
}


def highlight_excerpt(text, query):
    escaped = html.escape(clean_text(text))
    terms = []
    for token in re.findall(r"[A-Za-z0-9]{3,}", (query or "").lower()):
        if token not in SOURCE_STOPWORDS and token not in terms:
            terms.append(token)

    for term in sorted(terms, key=len, reverse=True):
        pattern = re.compile(rf"({re.escape(term)})", re.IGNORECASE)
        escaped = pattern.sub(r"<mark>\1</mark>", escaped)

    return escaped


def toggle_source_card(card_key):
    if st.session_state.selected_source_key == card_key:
        st.session_state.selected_source_key = None
    else:
        st.session_state.selected_source_key = card_key
    st.rerun()


def render_styles():
    st.markdown(
        """
        <style>
        :root {
            --bg-1: #081118;
            --bg-2: #101a21;
            --panel: rgba(12, 20, 27, 0.78);
            --line: rgba(255, 255, 255, 0.08);
            --line-strong: rgba(246, 180, 76, 0.35);
            --text: #f6f7f9;
            --muted: #a8b3bc;
            --accent: #f6b44c;
            --accent-soft: rgba(246, 180, 76, 0.18);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(97, 212, 198, 0.14), transparent 26%),
                radial-gradient(circle at top right, rgba(246, 180, 76, 0.14), transparent 28%),
                linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
            color: var(--text);
        }

        .main .block-container {
            max-width: 1120px;
            padding-top: 2.1rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(10, 17, 23, 0.95), rgba(15, 23, 30, 0.96));
            border-right: 1px solid var(--line);
        }

        .hero-shell {
            padding: 1.7rem 1.75rem;
            border: 1px solid var(--line);
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.015)),
                linear-gradient(180deg, rgba(8, 17, 24, 0.92), rgba(12, 20, 27, 0.92));
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.24);
            margin-bottom: 1rem;
        }

        .eyebrow {
            margin: 0 0 0.65rem 0;
            color: var(--accent);
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
        }

        .hero-title {
            margin: 0;
            color: var(--text);
            font-size: clamp(2rem, 5vw, 3.2rem);
            line-height: 1.02;
            font-weight: 700;
        }

        .hero-copy {
            margin: 0.9rem 0 0 0;
            max-width: 760px;
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.65;
        }

        .status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            margin-top: 1.2rem;
        }

        .status-card {
            min-width: 180px;
            padding: 0.85rem 1rem;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.035);
        }

        .status-label {
            display: block;
            color: var(--muted);
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .status-value {
            display: block;
            margin-top: 0.35rem;
            color: var(--text);
            font-size: 1rem;
            font-weight: 600;
        }

        .section-shell {
            margin-top: 1.3rem;
            padding: 1.25rem 1.25rem 0.25rem 1.25rem;
            border: 1px solid var(--line);
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(15, 23, 30, 0.9), rgba(10, 17, 23, 0.92));
        }

        .section-title {
            margin: 0;
            color: var(--text);
            font-size: 1.1rem;
            font-weight: 700;
        }

        .section-copy {
            margin: 0.4rem 0 1rem 0;
            color: var(--muted);
            line-height: 1.55;
        }

        .empty-shell {
            margin: 0.4rem 0 1rem 0;
            padding: 1rem 1.1rem;
            border-radius: 18px;
            border: 1px dashed var(--line);
            background: rgba(255, 255, 255, 0.025);
            color: var(--muted);
        }

        .error-shell {
            margin-top: 1rem;
            padding: 1rem 1.1rem;
            border-radius: 18px;
            border: 1px solid rgba(255, 126, 126, 0.24);
            background: rgba(111, 32, 32, 0.18);
        }

        .error-label {
            margin: 0;
            color: #ffd6d6;
            font-weight: 700;
        }

        .error-copy {
            margin: 0.4rem 0 0 0;
            color: #ffd6d6;
            line-height: 1.55;
        }

        .sidebar-note {
            padding: 0.9rem 1rem;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.03);
            color: var(--muted);
            line-height: 1.55;
        }

        .source-label {
            margin: 1rem 0 0.55rem 0;
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .source-summary {
            margin: -0.3rem 0 0.75rem 0.1rem;
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.55;
        }

        .source-detail-shell {
            margin: 0.25rem 0 1rem 0;
            padding: 1rem 1.05rem;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.025);
        }

        .source-detail-meta {
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .source-detail-copy {
            margin-top: 0.7rem;
            color: var(--text);
            line-height: 1.7;
        }

        .source-detail-copy mark {
            background: rgba(246, 180, 76, 0.28);
            color: var(--text);
            padding: 0.02rem 0.12rem;
            border-radius: 0.2rem;
        }

        div.stButton > button {
            width: 100%;
            min-height: 4.5rem;
            padding: 1rem 1.1rem;
            margin-bottom: 0.75rem;
            border-radius: 18px;
            border: 1px solid var(--line);
            background:
                linear-gradient(135deg, rgba(246, 180, 76, 0.08), rgba(97, 212, 198, 0.06)),
                linear-gradient(180deg, rgba(18, 28, 36, 0.98), rgba(12, 20, 27, 0.98));
            color: var(--text);
            font-size: 1rem;
            font-weight: 600;
            line-height: 1.4;
            text-align: left;
            transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
            box-shadow: 0 14px 35px rgba(0, 0, 0, 0.16);
        }

        div.stButton > button:hover {
            border-color: var(--line-strong);
            transform: translateY(-1px);
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.22);
        }

        div.stButton > button:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 0.15rem var(--accent-soft);
        }

        [data-testid="stChatMessage"] {
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 0.45rem 0.8rem;
            background: rgba(255, 255, 255, 0.03);
        }

        [data-testid="stChatInput"] {
            background: rgba(10, 17, 23, 0.94);
            border: 1px solid var(--line);
            border-radius: 20px;
        }

        [data-testid="stChatInput"] textarea {
            color: var(--text);
        }

        @media (max-width: 640px) {
            .main .block-container {
                padding-top: 1.35rem;
            }

            .hero-shell {
                padding: 1.25rem;
                border-radius: 22px;
            }

            .section-shell {
                padding: 1rem 1rem 0.1rem 1rem;
            }

            div.stButton > button {
                min-height: 4rem;
                font-size: 0.97rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    has_vectorstore = st.session_state.vectorstore is not None
    if has_vectorstore and st.session_state.document_count:
        document_status = f"{st.session_state.document_count} chunks indexed"
    elif has_vectorstore:
        document_status = "Knowledge base ready"
    else:
        document_status = "Waiting for upload"

    if st.session_state.suggested_questions:
        question_status = f"{len(st.session_state.suggested_questions)} suggestions ready"
    else:
        question_status = "Generated after upload"

    if st.session_state.chat_history:
        answer_count = len(
            [message for message in st.session_state.chat_history if message["role"] == "assistant"]
        )
        chat_status = f"{answer_count} answers delivered"
    else:
        chat_status = "No queries yet"

    st.markdown(
        f"""
        <section class="hero-shell">
            <p class="eyebrow">AI Research Workspace</p>
            <h1 class="hero-title">Turn your PDFs into clickable answers.</h1>
            <p class="hero-copy">
                Upload one or more documents, tap a suggested question, or ask your own.
                The app will search the indexed content and return a grounded answer with source context.
            </p>
            <div class="status-row">
                <div class="status-card">
                    <span class="status-label">Knowledge Base</span>
                    <span class="status-value">{document_status}</span>
                </div>
                <div class="status-card">
                    <span class="status-label">Suggested Questions</span>
                    <span class="status-value">{question_status}</span>
                </div>
                <div class="status-card">
                    <span class="status-label">Conversation</span>
                    <span class="status-value">{chat_status}</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_error_panel():
    last_error = st.session_state.last_error
    if not last_error:
        return

    st.markdown(
        f"""
        <div class="error-shell">
            <p class="error-label">Issue in {last_error["stage"]}</p>
            <p class="error-copy">{last_error["message"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Technical details", expanded=False):
        st.code(last_error["details"] or last_error["message"])


def prepare_uploaded_documents(files):
    all_docs = []
    for uploaded_file in files:
        documents = load_pdf(uploaded_file)
        all_docs.extend(semantic_chunk_documents(documents))

    vectorstore = create_vector_store(all_docs)
    questions = generate_questions(all_docs)
    return all_docs, vectorstore, questions


@st.cache_resource(show_spinner=False)
def get_graph():
    return build_graph()


def queue_suggested_question(question):
    st.session_state.pending_query = question
    clear_error()
    st.rerun()


def render_source_cards(message_index, message):
    source_cards = message.get("source_cards") or []
    if not source_cards:
        return

    st.markdown(
        f"""
        <div class="source-label">
            Sources Referenced · {len(source_cards)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    for source_index, source_card in enumerate(source_cards):
        card_key = f"source_{message_index}_{source_index}"
        if st.button(
            source_card["title"],
            key=f"source_button_{message_index}_{source_index}",
            use_container_width=True
        ):
            toggle_source_card(card_key)

        st.markdown(
            f'<div class="source-summary">{html.escape(source_card["summary"])}</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.selected_source_key == card_key:
            page = source_card.get("page", "N/A")
            if page in {None, "", "N/A"}:
                location = source_card["source"]
            else:
                location = f'{source_card["source"]} · Page {page}'

            highlighted = highlight_excerpt(source_card["excerpt"], message.get("query", ""))
            st.markdown(
                f"""
                <div class="source-detail-shell">
                    <div class="source-detail-meta">{html.escape(location)}</div>
                    <div class="source-detail-copy">{highlighted}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_chat_message(message, message_index):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            render_source_cards(message_index, message)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_error" not in st.session_state:
    st.session_state.last_error = None

if "vectorstore" not in st.session_state:
    try:
        st.session_state.vectorstore = load_vector_store()
    except Exception as exc:
        st.session_state.vectorstore = None
        record_error("loading the saved knowledge base", exc)

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "last_result" not in st.session_state:
    st.session_state.last_result = {}

if "upload_signature" not in st.session_state:
    st.session_state.upload_signature = None

if "document_count" not in st.session_state:
    st.session_state.document_count = 0

if "selected_source_key" not in st.session_state:
    st.session_state.selected_source_key = None


render_styles()

st.sidebar.markdown("## Knowledge Base")
st.sidebar.markdown(
    """
    <div class="sidebar-note">
        Add PDF documents to generate tailored questions and run source-backed answers from the same workspace.
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    current_signature = uploaded_file_signature(uploaded_files)
    if current_signature != st.session_state.upload_signature:
        try:
            with st.spinner("Indexing documents and drafting suggested questions..."):
                all_docs, vectorstore, questions = prepare_uploaded_documents(uploaded_files)
        except Exception as exc:
            record_error("preparing uploaded documents", exc)
            st.sidebar.error("Unable to process the uploaded PDFs. Check the error panel for details.")
        else:
            st.session_state.vectorstore = vectorstore
            st.session_state.suggested_questions = questions
            st.session_state.upload_signature = current_signature
            st.session_state.document_count = len(all_docs)
            st.session_state.chat_history = []
            st.session_state.last_result = {}
            st.session_state.selected_source_key = None
            clear_error()
            st.sidebar.success("Documents indexed and suggestions refreshed.")

if st.session_state.vectorstore is not None:
    st.sidebar.success("Knowledge base is ready.")

if st.session_state.suggested_questions:
    st.sidebar.caption(f"{len(st.session_state.suggested_questions)} suggested questions available")

render_header()

graph = None
try:
    graph = get_graph()
except Exception as exc:
    record_error("building the app workflow", exc)

render_error_panel()

st.markdown(
    """
    <section class="section-shell">
        <h2 class="section-title">Suggested Questions</h2>
        <p class="section-copy">
            Click any question to run it instantly through the same retrieval and answer pipeline as the chat input.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

if st.session_state.suggested_questions:
    for index, question in enumerate(st.session_state.suggested_questions):
        if st.button(question, key=f"suggested_question_{index}", use_container_width=True):
            queue_suggested_question(question)
else:
    st.markdown(
        """
        <div class="empty-shell">
            Upload PDF documents to generate tailored starter questions for this workspace.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <section class="section-shell">
        <h2 class="section-title">Conversation</h2>
        <p class="section-copy">
            Ask a custom question below or continue from one of the suggested prompts above.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

for message_index, message in enumerate(st.session_state.chat_history):
    render_chat_message(message, message_index)

typed_query = st.chat_input(
    "Ask about insights, metrics, risks, decisions, or next steps in your documents",
    key="chat_input"
)

query = st.session_state.pending_query or typed_query
if st.session_state.pending_query:
    st.session_state.pending_query = None

if query:
    user_message = {"role": "user", "content": query}
    st.session_state.chat_history.append(user_message)
    render_chat_message(user_message, len(st.session_state.chat_history) - 1)

    if st.session_state.vectorstore is None:
        response = "Upload at least one PDF before running a question."
        assistant_message = {"role": "assistant", "content": response, "query": query, "source_cards": []}
        st.session_state.chat_history.append(assistant_message)
        with st.chat_message("assistant"):
            st.warning(response)
    elif graph is None:
        response = "The workflow could not be initialized. Check the issue panel above before retrying."
        assistant_message = {"role": "assistant", "content": response, "query": query, "source_cards": []}
        st.session_state.chat_history.append(assistant_message)
        with st.chat_message("assistant"):
            st.error(response)
    else:
        state = {
            "query": query,
            "vectorstore": st.session_state.vectorstore,
            "chat_history": st.session_state.chat_history,
        }

        try:
            with st.spinner("Searching documents and drafting your answer..."):
                result = graph.invoke(state)
        except Exception as exc:
            record_error("running the query", exc)
            response = (
                "I hit an error while running that question. Check the issue panel above for the cause."
            )
            assistant_message = {"role": "assistant", "content": response, "query": query, "source_cards": []}
            st.session_state.chat_history.append(assistant_message)
            with st.chat_message("assistant"):
                st.error(response)
        else:
            clear_error()
            response = result.get(
                "response",
                "I could not find a supported answer in the uploaded documents.",
            )
            st.session_state.last_result = result
            assistant_message = {
                "role": "assistant",
                "content": response,
                "query": query,
                "source_cards": build_source_cards(result.get("documents", [])),
            }
            st.session_state.chat_history.append(assistant_message)
            render_chat_message(assistant_message, len(st.session_state.chat_history) - 1)
