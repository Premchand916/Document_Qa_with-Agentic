import html
import os
from pathlib import Path
import re
import sys
import traceback

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st

from agents.question_generator import generate_questions
from app.prompt_library import PROMPT_LIBRARY, get_prompt_categories
from ingestion.file_loader import SUPPORTED_FILE_TYPES, extract_tabular_assets, load_uploaded_file
from ingestion.semantic_chunker import semantic_chunk_documents
from langgraph_flow.graph_builder import build_graph
from vector_store.faiss_store import create_vector_store, load_vector_store

st.set_page_config(page_title="Document Intelligence Workspace", layout="wide")

ANSWER_MODES = [
    "Auto",
    "Executive Brief",
    "Analyst Deep Dive",
    "Comparison Matrix",
    "Action Plan",
    "Risk Review",
    "Data Highlights",
    "Research Synthesis",
]
AUDIENCE_OPTIONS = ["General", "Leadership", "Analyst", "Product", "Operations", "Sales"]
DEPTH_OPTIONS = ["Fast", "Balanced", "Comprehensive"]

MODE_DESCRIPTIONS = {
    "Auto": "Adaptive answer style based on the question and evidence.",
    "Executive Brief": "Short, strategic, decision-ready output.",
    "Analyst Deep Dive": "Detailed and evidence-rich response.",
    "Comparison Matrix": "Cross-file comparison with contrasts and overlaps.",
    "Action Plan": "Recommended actions, priorities, and dependencies.",
    "Risk Review": "Top risks, impact, and mitigations.",
    "Data Highlights": "Metrics, trends, and anomalies first.",
    "Research Synthesis": "Themes, tensions, and opportunities across files.",
}

AUDIOENCE_DESCRIPTIONS = {
    "General": "Clear plain-English answers.",
    "Leadership": "Implications and tradeoffs first.",
    "Analyst": "Evidence, assumptions, and caveats matter.",
    "Product": "User needs, opportunities, and priorities matter.",
    "Operations": "Execution, blockers, and workflows matter.",
    "Sales": "Value, objections, and proof points matter.",
}

DEPTH_DESCRIPTIONS = {
    "Fast": "Short and crisp.",
    "Balanced": "Balanced detail.",
    "Comprehensive": "Thorough and structured.",
}

SOURCE_STOPWORDS = {
    "about", "after", "again", "being", "below", "between", "could", "from",
    "have", "into", "just", "more", "most", "other", "same", "some", "such",
    "than", "that", "them", "then", "there", "these", "they", "this", "very",
    "what", "when", "where", "which", "while", "with", "would", "your"
}


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


def humanize_extension(extension):
    return extension.lstrip(".").upper()


def format_location_label(page):
    if page in {None, "", "N/A"}:
        return ""
    text = str(page).strip()
    if not text:
        return ""
    if text.isdigit():
        return f"Page {text}"
    if text.lower().startswith(("page", "slide", "sheet", "json", "txt", "md", "csv", "tsv")):
        return text
    return f"Page {text}"


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
        location = format_location_label(page)
        title = source_name if not location else f"{source_name} - {location}"
        cards.append(
            {
                "title": title,
                "summary": summarize_words(excerpt, max_words=20),
                "excerpt": excerpt,
            }
        )
    return cards


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
    st.session_state.selected_source_key = (
        None if st.session_state.selected_source_key == card_key else card_key
    )
    st.rerun()


def queue_prompt(prompt):
    st.session_state.pending_query = prompt
    clear_error()
    st.rerun()


def build_format_pills(file_types):
    if not file_types:
        return '<span class="pill">Formats appear after upload</span>'
    return "".join(f'<span class="pill">{html.escape(file_type)}</span>' for file_type in file_types)


def render_styles():
    st.markdown(
        """
        <style>
        :root { --bg-1:#081118; --bg-2:#101a21; --line:rgba(255,255,255,.08); --text:#f6f7f9; --muted:#a8b3bc; --accent:#f6b44c; }
        .stApp { background: radial-gradient(circle at top left, rgba(93,200,186,.16), transparent 26%), radial-gradient(circle at top right, rgba(246,180,76,.14), transparent 28%), linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%); color: var(--text); }
        .main .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(10,17,23,.95), rgba(15,23,30,.96)); border-right: 1px solid var(--line); }
        .hero-shell, .section-shell { border:1px solid var(--line); border-radius:24px; background: linear-gradient(180deg, rgba(15,23,30,.9), rgba(10,17,23,.92)); }
        .hero-shell { padding:1.8rem; margin-bottom:1rem; box-shadow:0 24px 70px rgba(0,0,0,.24); }
        .section-shell { padding:1.1rem 1.2rem .25rem 1.2rem; margin-top:1.2rem; }
        .eyebrow { margin:0 0 .65rem 0; color:var(--accent); font-size:.8rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
        .hero-title { margin:0; color:var(--text); font-size:clamp(2rem,5vw,3.05rem); line-height:1.04; font-weight:700; }
        .hero-copy, .section-copy, .library-meta, .mini-note { color:var(--muted); line-height:1.6; }
        .hero-copy { margin:.9rem 0 0 0; max-width:780px; font-size:1.02rem; }
        .section-title { margin:0; color:var(--text); font-size:1.08rem; font-weight:700; }
        .section-copy { margin:.4rem 0 1rem 0; }
        .status-row, .pill-row { display:flex; flex-wrap:wrap; gap:.75rem; margin-top:1rem; }
        .status-card, .config-card { min-width:185px; padding:.85rem 1rem; border-radius:18px; border:1px solid var(--line); background:rgba(255,255,255,.035); }
        .status-label, .source-label, .config-title { display:block; color:var(--muted); font-size:.78rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
        .status-value { display:block; margin-top:.35rem; color:var(--text); font-size:1rem; font-weight:600; }
        .pill { display:inline-flex; align-items:center; padding:.42rem .7rem; border-radius:999px; border:1px solid var(--line); background:rgba(255,255,255,.04); color:var(--text); font-size:.82rem; font-weight:600; }
        .sidebar-note, .empty-shell, .error-shell, .source-detail-shell { padding:1rem 1.05rem; border-radius:18px; border:1px solid var(--line); background:rgba(255,255,255,.03); color:var(--muted); line-height:1.55; }
        .empty-shell { border-style:dashed; background:rgba(255,255,255,.025); margin:.4rem 0 1rem 0; }
        .error-shell { border-color:rgba(255,126,126,.24); background:rgba(111,32,32,.18); margin-top:1rem; }
        .error-label, .source-detail-meta { color:#ffd6d6; font-weight:700; }
        .source-label { margin:1rem 0 .55rem 0; letter-spacing:.12em; }
        .source-summary { margin:-.3rem 0 .75rem .1rem; color:var(--muted); font-size:.98rem; line-height:1.55; }
        .source-detail-meta { color:var(--accent); font-size:.82rem; letter-spacing:.08em; text-transform:uppercase; }
        .source-detail-copy { margin-top:.7rem; color:var(--text); line-height:1.7; }
        .source-detail-copy mark { background:rgba(246,180,76,.28); color:var(--text); padding:.02rem .12rem; border-radius:.2rem; }
        div.stButton > button { width:100%; min-height:4.2rem; padding:1rem 1.05rem; margin-bottom:.72rem; border-radius:18px; border:1px solid var(--line); background:linear-gradient(135deg, rgba(246,180,76,.08), rgba(93,200,186,.06)), linear-gradient(180deg, rgba(18,28,36,.98), rgba(12,20,27,.98)); color:var(--text); font-size:.98rem; font-weight:600; line-height:1.4; text-align:left; box-shadow:0 14px 35px rgba(0,0,0,.16); }
        [data-testid="stChatMessage"] { border:1px solid var(--line); border-radius:22px; padding:.45rem .8rem; background:rgba(255,255,255,.03); }
        [data-testid="stChatInput"] { background:rgba(10,17,23,.94); border:1px solid var(--line); border-radius:20px; }
        [data-testid="stChatInput"] textarea { color:var(--text); }
        @media (max-width: 640px) { .main .block-container { padding-top:1.3rem; } .hero-shell { padding:1.2rem; } .section-shell { padding:1rem 1rem .1rem 1rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    has_vectorstore = st.session_state.vectorstore is not None
    document_status = (
        f"{st.session_state.document_count} chunks indexed"
        if has_vectorstore and st.session_state.document_count
        else "Knowledge base ready" if has_vectorstore else "Waiting for upload"
    )
    file_type_status = ", ".join(st.session_state.uploaded_file_types) if st.session_state.uploaded_file_types else "Upload files to see active formats"
    question_status = (
        f"{len(st.session_state.suggested_questions)} document prompts"
        if st.session_state.suggested_questions else "Generated after upload"
    )
    answer_count = len([message for message in st.session_state.chat_history if message["role"] == "assistant"])
    chat_status = f"{answer_count} answers delivered" if answer_count else "No queries yet"

    st.markdown(
        f"""
        <section class="hero-shell">
            <p class="eyebrow">Multi-Format AI Research Workspace</p>
            <h1 class="hero-title">Analyze PDFs, decks, sheets, CSVs, and text files in one flow.</h1>
            <p class="hero-copy">
                Upload mixed business files, use prompt-chained answering for different user needs,
                and explore 70+ prompt examples for summaries, comparisons, risks, actions,
                research synthesis, and data-heavy analysis.
            </p>
            <div class="status-row">
                <div class="status-card"><span class="status-label">Knowledge Base</span><span class="status-value">{document_status}</span></div>
                <div class="status-card"><span class="status-label">Active File Types</span><span class="status-value">{file_type_status}</span></div>
                <div class="status-card"><span class="status-label">Document Prompts</span><span class="status-value">{question_status}</span></div>
                <div class="status-card"><span class="status-label">Conversation</span><span class="status-value">{chat_status}</span></div>
            </div>
            <div class="pill-row">{build_format_pills(st.session_state.uploaded_file_types)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_error_panel():
    if not st.session_state.last_error:
        return
    last_error = st.session_state.last_error
    st.markdown(
        f"""
        <div class="error-shell">
            <p class="error-label">Issue in {last_error["stage"]}</p>
            <p>{last_error["message"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Technical details", expanded=False):
        st.code(last_error["details"] or last_error["message"])


def prepare_uploaded_documents(files):
    raw_documents = []
    tabular_assets = []
    uploaded_file_types = set()
    for uploaded_file in files:
        uploaded_file_types.add(humanize_extension(Path(uploaded_file.name).suffix.lower()))
        raw_documents.extend(load_uploaded_file(uploaded_file))
        tabular_assets.extend(extract_tabular_assets(uploaded_file))
    chunked_documents = semantic_chunk_documents(raw_documents)
    vectorstore = create_vector_store(chunked_documents)
    questions = generate_questions(raw_documents)
    source_count = len({doc.metadata.get("source", "Unknown") for doc in raw_documents})
    return (
        raw_documents,
        chunked_documents,
        vectorstore,
        questions,
        sorted(uploaded_file_types),
        source_count,
        tabular_assets,
    )


@st.cache_resource(show_spinner=False)
def get_graph():
    return build_graph()


def render_source_cards(message_index, message):
    source_cards = message.get("source_cards") or []
    if not source_cards:
        return
    st.markdown(f'<div class="source-label">Sources Referenced - {len(source_cards)}</div>', unsafe_allow_html=True)
    for source_index, source_card in enumerate(source_cards):
        card_key = f"source_{message_index}_{source_index}"
        if st.button(source_card["title"], key=f"source_button_{message_index}_{source_index}", use_container_width=True):
            toggle_source_card(card_key)
        st.markdown(f'<div class="source-summary">{html.escape(source_card["summary"])}</div>', unsafe_allow_html=True)
        if st.session_state.selected_source_key == card_key:
            highlighted = highlight_excerpt(source_card["excerpt"], message.get("query", ""))
            st.markdown(
                f"""
                <div class="source-detail-shell">
                    <div class="source-detail-meta">{html.escape(source_card["title"])}</div>
                    <div class="source-detail-copy">{highlighted}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_chat_message(message, message_index):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        response_meta = message.get("response_meta")
        if message["role"] == "assistant" and response_meta:
            st.caption(
                f"Mode: {response_meta.get('answer_mode', 'Auto')} | "
                f"Intent: {response_meta.get('intent', 'document_qa')} | "
                f"Need: {response_meta.get('user_need', 'Evidence-based answer')}"
            )
        if message["role"] == "assistant":
            render_source_cards(message_index, message)


def render_prompt_library():
    st.markdown(
        """
        <section class="section-shell">
            <h2 class="section-title">Prompt Library</h2>
            <p class="section-copy">Explore 72 curated prompt examples across different user needs and output types.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    selected_category = st.selectbox("Browse examples by user need", get_prompt_categories(), key="prompt_category")
    category_data = PROMPT_LIBRARY[selected_category]
    st.markdown(f'<div class="library-meta">{html.escape(category_data["description"])}</div>', unsafe_allow_html=True)
    left_column, right_column = st.columns(2)
    for index, prompt in enumerate(category_data["prompts"]):
        target = left_column if index % 2 == 0 else right_column
        with target:
            if st.button(prompt, key=f"library_prompt_{selected_category}_{index}", use_container_width=True):
                queue_prompt(prompt)


def render_document_prompts():
    st.markdown(
        """
        <section class="section-shell">
            <h2 class="section-title">Document-Specific Prompts</h2>
            <p class="section-copy">These suggestions are generated from the uploaded files and balanced across sources.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.suggested_questions:
        for index, question in enumerate(st.session_state.suggested_questions):
            if st.button(question, key=f"suggested_question_{index}", use_container_width=True):
                queue_prompt(question)
    else:
        st.markdown(
            '<div class="empty-shell">Upload supported files to generate tailored prompts from your own content.</div>',
            unsafe_allow_html=True,
        )


def render_answer_blueprint():
    st.markdown(
        """
        <section class="section-shell">
            <h2 class="section-title">Answer Blueprint</h2>
            <p class="section-copy">Prompt chaining uses your selected answer mode, audience, and depth to shape the final response.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(3)
    blocks = [
        ("Answer Mode", st.session_state.answer_mode, MODE_DESCRIPTIONS[st.session_state.answer_mode]),
        ("Audience", st.session_state.audience, AUDIOENCE_DESCRIPTIONS[st.session_state.audience]),
        ("Depth", st.session_state.response_depth, DEPTH_DESCRIPTIONS[st.session_state.response_depth]),
    ]
    for column, (title, value, copy) in zip(columns, blocks):
        with column:
            st.markdown(
                f"""
                <div class="config-card">
                    <p class="config-title">{html.escape(title)}</p>
                    <p class="mini-note"><strong>{html.escape(value)}</strong><br>{html.escape(copy)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def initialize_state():
    defaults = {
        "chat_history": [],
        "last_error": None,
        "vectorstore": None,
        "suggested_questions": [],
        "pending_query": None,
        "last_result": {},
        "upload_signature": None,
        "document_count": 0,
        "uploaded_file_types": [],
        "source_count": 0,
        "selected_source_key": None,
        "tabular_assets": [],
        "answer_mode": "Auto",
        "audience": "General",
        "response_depth": "Balanced",
        "prompt_category": get_prompt_categories()[0],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.vectorstore is None:
        try:
            st.session_state.vectorstore = load_vector_store()
        except Exception as exc:
            st.session_state.vectorstore = None
            record_error("loading the saved knowledge base", exc)


initialize_state()
render_styles()

st.sidebar.markdown("## Knowledge Base")
st.sidebar.markdown(
    """
    <div class="sidebar-note">
        Upload mixed file types, generate tailored prompts, and answer them with prompt-chained reasoning across documents, slides, and data tables.
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("### Supported Formats")
st.sidebar.caption(", ".join(sorted(label for label in SUPPORTED_FILE_TYPES.values())))
st.sidebar.markdown("### Answer Controls")
st.session_state.answer_mode = st.sidebar.selectbox("Answer mode", ANSWER_MODES, index=ANSWER_MODES.index(st.session_state.answer_mode))
st.sidebar.caption(MODE_DESCRIPTIONS[st.session_state.answer_mode])
st.session_state.audience = st.sidebar.selectbox("Audience", AUDIENCE_OPTIONS, index=AUDIENCE_OPTIONS.index(st.session_state.audience))
st.sidebar.caption(AUDIOENCE_DESCRIPTIONS[st.session_state.audience])
st.session_state.response_depth = st.sidebar.select_slider("Depth", options=DEPTH_OPTIONS, value=st.session_state.response_depth)
st.sidebar.caption(DEPTH_DESCRIPTIONS[st.session_state.response_depth])

uploaded_files = st.sidebar.file_uploader(
    "Upload business files",
    type=[extension.lstrip(".") for extension in SUPPORTED_FILE_TYPES.keys()],
    accept_multiple_files=True,
)

if uploaded_files:
    current_signature = uploaded_file_signature(uploaded_files)
    if current_signature != st.session_state.upload_signature:
        try:
            with st.spinner("Reading files, chunking content, building prompts, and indexing the workspace..."):
                (
                    raw_docs,
                    chunked_docs,
                    vectorstore,
                    questions,
                    file_types,
                    source_count,
                    tabular_assets,
                ) = prepare_uploaded_documents(uploaded_files)
        except Exception as exc:
            record_error("preparing uploaded documents", exc)
            st.sidebar.error("Unable to process the uploaded files. Check the issue panel for details.")
        else:
            st.session_state.vectorstore = vectorstore
            st.session_state.suggested_questions = questions
            st.session_state.upload_signature = current_signature
            st.session_state.document_count = len(chunked_docs)
            st.session_state.uploaded_file_types = file_types
            st.session_state.source_count = source_count
            st.session_state.tabular_assets = tabular_assets
            st.session_state.chat_history = []
            st.session_state.last_result = {}
            st.session_state.selected_source_key = None
            clear_error()
            st.sidebar.success(f"Indexed {len(chunked_docs)} chunks from {source_count} files across {', '.join(file_types)}.")

if st.session_state.vectorstore is not None:
    st.sidebar.success("Knowledge base is ready.")
if st.session_state.uploaded_file_types:
    st.sidebar.caption("Active formats: " + ", ".join(st.session_state.uploaded_file_types))
if st.session_state.suggested_questions:
    st.sidebar.caption(f"{len(st.session_state.suggested_questions)} document prompts available")

render_header()
graph = None
try:
    graph = get_graph()
except Exception as exc:
    record_error("building the app workflow", exc)
render_error_panel()

render_answer_blueprint()
render_prompt_library()
render_document_prompts()

st.markdown(
    """
    <section class="section-shell">
        <h2 class="section-title">Conversation</h2>
        <p class="section-copy">Ask about PDFs, PowerPoints, Excel workbooks, CSV files, markdown, JSON, or mixed uploads in one place.</p>
    </section>
    """,
    unsafe_allow_html=True,
)
for message_index, message in enumerate(st.session_state.chat_history):
    render_chat_message(message, message_index)

typed_query = st.chat_input(
    "Ask for summaries, comparisons, metrics, risks, requirements, action plans, or cross-file insights",
    key="chat_input",
)
query = st.session_state.pending_query or typed_query
if st.session_state.pending_query:
    st.session_state.pending_query = None

if query:
    user_message = {"role": "user", "content": query}
    st.session_state.chat_history.append(user_message)
    render_chat_message(user_message, len(st.session_state.chat_history) - 1)
    if st.session_state.vectorstore is None:
        response = "Upload at least one supported file before running a question."
        assistant_message = {
            "role": "assistant",
            "content": response,
            "query": query,
            "source_cards": [],
            "response_meta": {"answer_mode": st.session_state.answer_mode, "intent": "document_qa", "user_need": "Upload files first."},
        }
        st.session_state.chat_history.append(assistant_message)
        with st.chat_message("assistant"):
            st.warning(response)
    elif graph is None:
        response = "The workflow could not be initialized. Check the issue panel above before retrying."
        assistant_message = {
            "role": "assistant",
            "content": response,
            "query": query,
            "source_cards": [],
            "response_meta": {"answer_mode": st.session_state.answer_mode, "intent": "document_qa", "user_need": "Resolve the workflow issue."},
        }
        st.session_state.chat_history.append(assistant_message)
        with st.chat_message("assistant"):
            st.error(response)
    else:
        state = {
            "query": query,
            "vectorstore": st.session_state.vectorstore,
            "chat_history": st.session_state.chat_history,
            "answer_mode": st.session_state.answer_mode,
            "audience": st.session_state.audience,
            "response_depth": st.session_state.response_depth,
            "uploaded_file_types": st.session_state.uploaded_file_types,
            "tabular_assets": st.session_state.tabular_assets,
        }
        try:
            with st.spinner("Planning the answer, retrieving evidence, and drafting a response..."):
                result = graph.invoke(state)
        except Exception as exc:
            record_error("running the query", exc)
            response = "I hit an error while running that question. Check the issue panel above for the cause."
            assistant_message = {
                "role": "assistant",
                "content": response,
                "query": query,
                "source_cards": [],
                "response_meta": {"answer_mode": st.session_state.answer_mode, "intent": "document_qa", "user_need": "Recover from the execution error."},
            }
            st.session_state.chat_history.append(assistant_message)
            with st.chat_message("assistant"):
                st.error(response)
        else:
            clear_error()
            response = result.get("response", "I could not find a supported answer in the uploaded documents.")
            st.session_state.last_result = result
            assistant_message = {
                "role": "assistant",
                "content": response,
                "query": query,
                "source_cards": build_source_cards(result.get("documents", [])),
                "response_meta": result.get("response_plan", {}),
            }
            st.session_state.chat_history.append(assistant_message)
            render_chat_message(assistant_message, len(st.session_state.chat_history) - 1)
