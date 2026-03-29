import html
import os
from datetime import datetime
from pathlib import Path
import re
import sys
import traceback

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st

from agents.orchestrator_agent import get_active_provider
from agents.question_generator import generate_questions
from app.prompt_library import PROMPT_LIBRARY, get_prompt_categories
from ingestion.file_loader import SUPPORTED_FILE_TYPES, extract_tabular_assets, load_uploaded_file
from ingestion.semantic_chunker import semantic_chunk_documents
from langgraph_flow.graph_builder import build_graph
from utils.prompt_skill import build_prompted_query, get_prompt_skill
from vector_Store.faiss_store import create_vector_store, load_vector_store

st.set_page_config(page_title="NotebookLM — Document Intelligence", layout="wide")

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

AUDIENCE_DESCRIPTIONS = {
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

FILE_ICONS = {
    "pdf": "📄",
    "xlsx": "📊",
    "xlsm": "📊",
    "csv": "📊",
    "tsv": "📊",
    "pptx": "📑",
    "txt": "📝",
    "md": "📝",
    "json": "🔧",
}

SOURCE_STOPWORDS = {
    "about", "after", "again", "being", "below", "between", "could", "from",
    "have", "into", "just", "more", "most", "other", "same", "some", "such",
    "than", "that", "them", "then", "there", "these", "they", "this", "very",
    "what", "when", "where", "which", "while", "with", "would", "your",
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
    if page in {None, "", "N/A", "Web"}:
        return str(page) if page == "Web" else ""
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
    for document in documents or []:
        metadata = getattr(document, "metadata", {}) or {}
        content_type = metadata.get("content_type", "")
        source_name = metadata.get("source", "Unknown document")
        page = metadata.get("page", "N/A")
        excerpt = clean_text(getattr(document, "page_content", ""))
        if not excerpt:
            continue
        dedupe_key = (source_name, page, excerpt[:160])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        if content_type == "web_result":
            title = metadata.get("title") or source_name
            cards.append({
                "title": title,
                "summary": summarize_words(excerpt, max_words=20),
                "excerpt": excerpt,
                "is_web": True,
                "url": metadata.get("url", source_name),
            })
        else:
            location = format_location_label(page)
            title = source_name if not location else f"{source_name} — {location}"
            cards.append({
                "title": title,
                "summary": summarize_words(excerpt, max_words=20),
                "excerpt": excerpt,
                "is_web": False,
                "url": "",
            })
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


def prepare_uploaded_documents(files):
    raw_documents = []
    tabular_assets = []
    uploaded_file_types = set()
    for uploaded_file in files:
        ext = Path(uploaded_file.name).suffix.lower()
        uploaded_file_types.add(humanize_extension(ext))
        raw_documents.extend(load_uploaded_file(uploaded_file))
        tabular_assets.extend(extract_tabular_assets(uploaded_file))
    chunked_documents = semantic_chunk_documents(raw_documents)
    vectorstore = None
    try:
        vectorstore = create_vector_store(chunked_documents)
    except Exception:
        if not tabular_assets:
            raise
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

    web_cards = [c for c in source_cards if c.get("is_web")]
    doc_cards = [c for c in source_cards if not c.get("is_web")]

    if doc_cards:
        st.markdown(
            f'<div style="color:#a8b3bc;font-size:.78rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin:.8rem 0 .4rem 0;">Sources — {len(doc_cards)}</div>',
            unsafe_allow_html=True,
        )
        for source_index, source_card in enumerate(doc_cards):
            card_key = f"source_{message_index}_{source_index}"
            if st.button(source_card["title"], key=f"src_btn_{message_index}_{source_index}", use_container_width=True):
                toggle_source_card(card_key)
            st.markdown(
                f'<div style="color:#a8b3bc;font-size:.88rem;margin:-.25rem 0 .6rem .05rem;">{html.escape(source_card["summary"])}</div>',
                unsafe_allow_html=True,
            )
            if st.session_state.selected_source_key == card_key:
                highlighted = highlight_excerpt(source_card["excerpt"], message.get("query", ""))
                st.markdown(
                    f"""
                    <div style="padding:.9rem 1rem;border-radius:14px;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.03);margin-bottom:.6rem;">
                        <div style="color:#f6b44c;font-size:.78rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;">{html.escape(source_card["title"])}</div>
                        <div style="margin-top:.6rem;color:#f6f7f9;line-height:1.7;">{highlighted}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if web_cards:
        st.markdown(
            '<div style="color:#5dc8ba;font-size:.78rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin:.8rem 0 .4rem 0;">🌐 Web Sources</div>',
            unsafe_allow_html=True,
        )
        for web_card in web_cards:
            url = web_card.get("url", "")
            title = web_card["title"]
            st.markdown(
                f'<div style="margin-bottom:.3rem;"><a href="{html.escape(url)}" target="_blank" style="color:#5dc8ba;font-size:.9rem;text-decoration:none;">🌐 {html.escape(title)}</a></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="color:#a8b3bc;font-size:.85rem;margin-bottom:.6rem;">{html.escape(web_card["summary"])}</div>',
                unsafe_allow_html=True,
            )


def render_chat_message(message, message_index):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        response_meta = message.get("response_meta")
        if message["role"] == "assistant" and response_meta:
            mode = response_meta.get("answer_mode", "Auto")
            intent = response_meta.get("intent", "document_qa")
            need = response_meta.get("user_need", "Evidence-based answer")
            st.caption(f"Mode: {mode} | Intent: {intent} | Need: {need}")
        if message["role"] == "assistant":
            render_source_cards(message_index, message)


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
        "use_web_search": False,
        "notes": [],
        "uploaded_file_list": [],
        "llm_provider": get_active_provider(),
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


def render_styles():
    st.markdown(
        """
        <style>
        :root {
            --bg-1: #081118; --bg-2: #101a21;
            --line: rgba(255,255,255,.08); --text: #f6f7f9;
            --muted: #a8b3bc; --accent: #f6b44c; --teal: #5dc8ba;
        }
        .stApp {
            background: radial-gradient(circle at top left, rgba(93,200,186,.14), transparent 26%),
                        radial-gradient(circle at top right, rgba(246,180,76,.12), transparent 28%),
                        linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
            color: var(--text);
        }
        /* Hide default sidebar */
        [data-testid="stSidebar"] { display: none !important; }
        .main .block-container { max-width: 100% !important; padding: 1.2rem 1.5rem 2rem 1.5rem; }

        /* Panel cards */
        .panel-card {
            border: 1px solid var(--line); border-radius: 20px;
            background: linear-gradient(180deg, rgba(15,23,30,.92), rgba(10,17,23,.95));
            padding: 1.1rem 1rem; height: 100%;
        }
        .panel-title {
            color: var(--accent); font-size: .75rem; font-weight: 700;
            letter-spacing: .16em; text-transform: uppercase; margin: 0 0 .8rem 0;
        }

        /* Source item in left panel */
        .source-item {
            display: flex; align-items: center; gap: .5rem;
            padding: .45rem .6rem; border-radius: 10px;
            background: rgba(255,255,255,.03); border: 1px solid var(--line);
            margin-bottom: .4rem; font-size: .9rem; color: var(--text);
            word-break: break-all;
        }

        /* Web search toggle */
        .web-toggle-on {
            display: inline-flex; align-items: center; gap: .4rem;
            padding: .4rem .85rem; border-radius: 999px;
            background: rgba(93,200,186,.18); border: 1px solid rgba(93,200,186,.4);
            color: var(--teal); font-size: .85rem; font-weight: 700; cursor: pointer;
        }
        .web-toggle-off {
            display: inline-flex; align-items: center; gap: .4rem;
            padding: .4rem .85rem; border-radius: 999px;
            background: rgba(255,255,255,.04); border: 1px solid var(--line);
            color: var(--muted); font-size: .85rem; font-weight: 600; cursor: pointer;
        }

        /* Note card */
        .note-card {
            padding: .75rem .85rem; border-radius: 14px;
            border: 1px solid var(--line); background: rgba(255,255,255,.03);
            margin-bottom: .6rem;
        }
        .note-source { color: var(--teal); font-size: .75rem; font-weight: 700;
            letter-spacing: .06em; text-transform: uppercase; margin-bottom: .3rem; }
        .note-text { color: var(--text); font-size: .88rem; line-height: 1.55; }
        .note-ts { color: var(--muted); font-size: .75rem; margin-top: .3rem; }

        /* Chat bubbles */
        [data-testid="stChatMessage"] {
            border: 1px solid var(--line); border-radius: 18px;
            padding: .4rem .7rem; background: rgba(255,255,255,.025);
        }
        [data-testid="stChatInput"] {
            background: rgba(10,17,23,.94); border: 1px solid var(--line); border-radius: 18px;
        }
        [data-testid="stChatInput"] textarea { color: var(--text); }

        /* Prompt buttons */
        div.stButton > button {
            width: 100%; padding: .65rem .85rem; border-radius: 14px;
            border: 1px solid var(--line);
            background: linear-gradient(135deg, rgba(246,180,76,.07), rgba(93,200,186,.05)),
                        linear-gradient(180deg, rgba(18,28,36,.98), rgba(12,20,27,.98));
            color: var(--text); font-size: .88rem; font-weight: 600; text-align: left;
            box-shadow: 0 8px 24px rgba(0,0,0,.14); margin-bottom: .45rem;
        }

        /* Top bar */
        .top-bar {
            display: flex; align-items: center; justify-content: space-between;
            padding: .7rem 1.2rem; border-radius: 18px; margin-bottom: 1.1rem;
            background: linear-gradient(180deg, rgba(15,23,30,.92), rgba(10,17,23,.95));
            border: 1px solid var(--line);
        }
        .top-bar-title { color: var(--text); font-size: 1.25rem; font-weight: 700; margin: 0; }
        .top-bar-sub { color: var(--muted); font-size: .83rem; margin: 0; }
        .status-pill {
            display: inline-flex; align-items: center; gap: .35rem;
            padding: .3rem .7rem; border-radius: 999px; font-size: .78rem; font-weight: 600;
            border: 1px solid var(--line); background: rgba(255,255,255,.04); color: var(--muted);
        }
        .status-pill.ready { border-color: rgba(93,200,186,.35); background: rgba(93,200,186,.1); color: var(--teal); }

        mark { background: rgba(246,180,76,.28); color: var(--text); padding: .02rem .12rem; border-radius: .2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Bootstrap ────────────────────────────────────────────────────────────────
initialize_state()
render_styles()

graph = None
try:
    graph = get_graph()
except Exception as exc:
    record_error("building the app workflow", exc)

# ── Top Bar ──────────────────────────────────────────────────────────────────
kb_ready = st.session_state.vectorstore is not None
data_ready = bool(st.session_state.tabular_assets)
web_on = st.session_state.use_web_search
status_class = "status-pill ready" if kb_ready or data_ready or web_on else "status-pill"
status_text = (
    "Knowledge base ready" if kb_ready else
    ("Structured data ready" if data_ready else
     ("Web search active" if web_on else "Upload files to begin"))
)

answer_count = len([m for m in st.session_state.chat_history if m["role"] == "assistant"])

st.markdown(
    f"""
    <div class="top-bar">
        <div>
            <p class="top-bar-title">📓 NotebookLM — Document Intelligence</p>
            <p class="top-bar-sub">Multi-format AI research workspace · PDFs, slides, spreadsheets, CSV, text &amp; web</p>
        </div>
        <div style="display:flex;gap:.5rem;flex-wrap:wrap;">
            <span class="{status_class}">{html.escape(status_text)}</span>
            <span class="status-pill">{answer_count} answers</span>
            <span class="status-pill">{len(st.session_state.notes)} notes</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Error Panel ──────────────────────────────────────────────────────────────
if st.session_state.last_error:
    err = st.session_state.last_error
    st.error(f"Issue in **{err['stage']}**: {err['message']}")
    with st.expander("Technical details", expanded=False):
        st.code(err["details"] or err["message"])

# ── 3-Column Layout ──────────────────────────────────────────────────────────
col_sources, col_chat, col_studio = st.columns([1.15, 2.3, 1.15], gap="medium")

# ════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — Sources
# ════════════════════════════════════════════════════════════════════════════
with col_sources:
    st.markdown('<div class="panel-title">Sources</div>', unsafe_allow_html=True)

    # File uploader
    uploaded_files = st.file_uploader(
        "Add source files",
        type=[ext.lstrip(".") for ext in SUPPORTED_FILE_TYPES.keys()],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        current_sig = uploaded_file_signature(uploaded_files)
        if current_sig != st.session_state.upload_signature:
            try:
                with st.spinner("Indexing files…"):
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
                st.error("Could not process files.")
            else:
                st.session_state.vectorstore = vectorstore
                st.session_state.suggested_questions = questions
                st.session_state.upload_signature = current_sig
                st.session_state.document_count = len(chunked_docs)
                st.session_state.uploaded_file_types = file_types
                st.session_state.source_count = source_count
                st.session_state.tabular_assets = tabular_assets
                st.session_state.chat_history = []
                st.session_state.last_result = {}
                st.session_state.selected_source_key = None
                # Build file list for display
                st.session_state.uploaded_file_list = [
                    {"name": f.name, "ext": Path(f.name).suffix.lstrip(".").lower()}
                    for f in uploaded_files
                ]
                clear_error()
                st.rerun()

    # Source list
    if st.session_state.uploaded_file_list:
        for file_info in st.session_state.uploaded_file_list:
            icon = FILE_ICONS.get(file_info["ext"], "📄")
            st.markdown(
                f'<div class="source-item">{icon} <span>{html.escape(file_info["name"])}</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="color:#a8b3bc;font-size:.85rem;padding:.5rem 0;">No sources added yet.</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.tabular_assets and st.session_state.vectorstore is None:
        st.markdown(
            '<div style="color:#5dc8ba;font-size:.8rem;padding:.3rem 0 .1rem 0;">Structured-data mode is available. Semantic search is unavailable until the embedding model/index can be built.</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Answer controls
    st.markdown('<div style="color:#a8b3bc;font-size:.78rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.5rem;">Answer Controls</div>', unsafe_allow_html=True)
    st.session_state.answer_mode = st.selectbox(
        "Answer mode", ANSWER_MODES,
        index=ANSWER_MODES.index(st.session_state.answer_mode),
        key="answer_mode_select",
        label_visibility="collapsed",
    )
    st.caption(MODE_DESCRIPTIONS[st.session_state.answer_mode])

    st.session_state.audience = st.selectbox(
        "Audience", AUDIENCE_OPTIONS,
        index=AUDIENCE_OPTIONS.index(st.session_state.audience),
        key="audience_select",
        label_visibility="collapsed",
    )
    st.caption(AUDIENCE_DESCRIPTIONS[st.session_state.audience])

    st.session_state.response_depth = st.select_slider(
        "Depth", options=DEPTH_OPTIONS,
        value=st.session_state.response_depth,
        key="depth_slider",
        label_visibility="collapsed",
    )
    st.caption(DEPTH_DESCRIPTIONS[st.session_state.response_depth])

    st.divider()

    # Web search toggle
    web_label = "🌐 Web Search: ON" if st.session_state.use_web_search else "🌐 Web Search: OFF"
    web_help = "Click to disable web search and return to document mode." if st.session_state.use_web_search else "Click to enable Tavily web search. Documents will be bypassed."
    if st.button(web_label, key="web_toggle_btn", help=web_help, use_container_width=True):
        st.session_state.use_web_search = not st.session_state.use_web_search
        st.rerun()

    if st.session_state.use_web_search:
        st.markdown(
            '<div style="color:#5dc8ba;font-size:.78rem;margin-top:.3rem;">Web mode active — answers come from Tavily.</div>',
            unsafe_allow_html=True,
        )
        if not os.getenv("TAVILY_API_KEY"):
            st.markdown(
                '<div style="color:#f6b44c;font-size:.76rem;margin-top:.2rem;">Add `TAVILY_API_KEY` to your `.env` file to enable live web search.</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # LLM Provider toggle (Gemini ↔ Ollama)
    st.markdown(
        '<div style="color:#a8b3bc;font-size:.78rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem;">LLM Provider</div>',
        unsafe_allow_html=True,
    )
    provider_options = ["gemini", "ollama"]
    current_provider = st.session_state.llm_provider
    new_provider = st.radio(
        "LLM provider",
        options=provider_options,
        index=provider_options.index(current_provider) if current_provider in provider_options else 0,
        format_func=lambda p: "🤖 Google Gemini" if p == "gemini" else "🦙 Ollama (Local)",
        key="llm_provider_radio",
        label_visibility="collapsed",
        horizontal=True,
    )
    if new_provider != current_provider:
        st.session_state.llm_provider = new_provider
        os.environ["LLM_PROVIDER"] = new_provider
        st.rerun()

    os.environ["LLM_PROVIDER"] = st.session_state.llm_provider
    active_ollama_model = os.getenv("OLLAMA_ACTIVE_MODEL") or os.getenv("OLLAMA_MODEL", "llama3.2")
    prompt_skill = get_prompt_skill()

    if st.session_state.llm_provider == "ollama":
        st.markdown(
            f'<div style="color:#f6b44c;font-size:.78rem;margin-top:.3rem;">🦙 Ollama active — model: {html.escape(active_ollama_model)}. If a large model runs out of RAM, the app will try a smaller installed local model automatically.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="color:#a8b3bc;font-size:.78rem;margin-top:.3rem;">🤖 Gemini active — uses your GOOGLE_API_KEY.</div>',
            unsafe_allow_html=True,
        )

    if prompt_skill.get("path"):
        st.markdown(
            f'<div style="color:#a8b3bc;font-size:.78rem;margin-top:.3rem;">Prompt profile active — {html.escape(Path(prompt_skill["path"]).name)}.</div>',
            unsafe_allow_html=True,
        )

    # Document prompts (collapsible)
    if st.session_state.suggested_questions:
        with st.expander("Document prompts", expanded=False):
            for idx, question in enumerate(st.session_state.suggested_questions):
                if st.button(question, key=f"doc_q_{idx}", use_container_width=True):
                    queue_prompt(question)


# ════════════════════════════════════════════════════════════════════════════
# CENTER PANEL — Chat
# ════════════════════════════════════════════════════════════════════════════
with col_chat:
    st.markdown('<div class="panel-title">Chat</div>', unsafe_allow_html=True)

    # Prompt library expander
    with st.expander("Browse prompt library", expanded=False):
        selected_category = st.selectbox(
            "Category", get_prompt_categories(),
            key="prompt_category_select",
            label_visibility="collapsed",
        )
        category_data = PROMPT_LIBRARY[selected_category]
        st.caption(category_data["description"])
        lib_left, lib_right = st.columns(2)
        for idx, prompt in enumerate(category_data["prompts"]):
            target = lib_left if idx % 2 == 0 else lib_right
            with target:
                if st.button(prompt, key=f"lib_{selected_category}_{idx}", use_container_width=True):
                    queue_prompt(prompt)

    # Chat history container
    chat_container = st.container(height=520)
    with chat_container:
        for msg_idx, message in enumerate(st.session_state.chat_history):
            render_chat_message(message, msg_idx)

    # Web search status indicator above chat input
    if st.session_state.use_web_search:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:.4rem;margin:.4rem 0 .2rem 0;"><span style="color:#5dc8ba;font-size:.82rem;font-weight:700;">🌐 Web Search is ON</span><span style="color:#a8b3bc;font-size:.78rem;">— answers will come from Tavily</span></div>',
            unsafe_allow_html=True,
        )

    typed_query = st.chat_input(
        "Ask anything about your sources, or search the web…",
        key="chat_input",
    )
    query = st.session_state.pending_query or typed_query
    if st.session_state.pending_query:
        st.session_state.pending_query = None

    if query:
        user_message = {"role": "user", "content": query}
        st.session_state.chat_history.append(user_message)

        can_run = (
            st.session_state.use_web_search
            or st.session_state.vectorstore is not None
            or bool(st.session_state.tabular_assets)
        )

        if not can_run:
            response = "Upload at least one file, or enable Web Search to ask questions."
            assistant_message = {
                "role": "assistant", "content": response, "query": query,
                "source_cards": [],
                "response_meta": {"answer_mode": st.session_state.answer_mode, "intent": "document_qa", "user_need": "Upload files or enable web search."},
            }
            st.session_state.chat_history.append(assistant_message)
            st.rerun()
        elif graph is None:
            response = "The workflow could not be initialized. Check the issue panel above."
            assistant_message = {
                "role": "assistant", "content": response, "query": query,
                "source_cards": [],
                "response_meta": {"answer_mode": st.session_state.answer_mode, "intent": "document_qa", "user_need": "Resolve the workflow issue."},
            }
            st.session_state.chat_history.append(assistant_message)
            st.rerun()
        else:
            prompt_query = build_prompted_query(
                query,
                {
                    "vectorstore": st.session_state.vectorstore,
                    "chat_history": st.session_state.chat_history,
                    "answer_mode": st.session_state.answer_mode,
                    "audience": st.session_state.audience,
                    "response_depth": st.session_state.response_depth,
                    "uploaded_file_types": st.session_state.uploaded_file_types,
                    "tabular_assets": st.session_state.tabular_assets,
                    "use_web_search": st.session_state.use_web_search,
                },
            )
            state = {
                "query": query,
                "prompt_query": prompt_query,
                "web_search_query": query,
                "prompt_skill_path": prompt_skill.get("path", ""),
                "vectorstore": st.session_state.vectorstore,
                "chat_history": st.session_state.chat_history,
                "answer_mode": st.session_state.answer_mode,
                "audience": st.session_state.audience,
                "response_depth": st.session_state.response_depth,
                "uploaded_file_types": st.session_state.uploaded_file_types,
                "tabular_assets": st.session_state.tabular_assets,
                "use_web_search": st.session_state.use_web_search,
            }
            try:
                with st.spinner("Thinking…"):
                    result = graph.invoke(state)
            except Exception as exc:
                record_error("running the query", exc)
                response = "I hit an error while running that question. Check the issue panel above."
                assistant_message = {
                    "role": "assistant", "content": response, "query": query,
                    "source_cards": [],
                    "response_meta": {"answer_mode": st.session_state.answer_mode, "intent": "document_qa", "user_need": "Recover from the error."},
                }
                st.session_state.chat_history.append(assistant_message)
                st.rerun()
            else:
                clear_error()
                response = result.get("response", "I could not find a supported answer.")
                st.session_state.last_result = result
                assistant_message = {
                    "role": "assistant",
                    "content": response,
                    "query": query,
                    "source_cards": build_source_cards(result.get("documents", [])),
                    "response_meta": result.get("response_plan", {}),
                }
                st.session_state.chat_history.append(assistant_message)
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — Studio
# ════════════════════════════════════════════════════════════════════════════
with col_studio:
    st.markdown('<div class="panel-title">Studio</div>', unsafe_allow_html=True)

    # Save note from last response
    last_result = st.session_state.last_result
    last_response = last_result.get("response", "")
    last_source = last_result.get("source", "")

    if last_response:
        if st.button("📌 Save last response as note", key="save_note_btn", use_container_width=True):
            st.session_state.notes.append({
                "content": last_response,
                "source": last_source or "Unknown source",
                "timestamp": datetime.now().strftime("%b %d, %H:%M"),
            })
            st.rerun()
    else:
        st.markdown(
            '<div style="color:#a8b3bc;font-size:.83rem;margin-bottom:.6rem;">Ask a question to save a note.</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Notes list
    notes = st.session_state.notes
    if notes:
        st.markdown(
            f'<div style="color:#a8b3bc;font-size:.78rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.6rem;">Saved Notes ({len(notes)})</div>',
            unsafe_allow_html=True,
        )
        for note_idx, note in enumerate(reversed(notes)):
            actual_idx = len(notes) - 1 - note_idx
            src_escaped = html.escape(note.get("source", ""))
            text_escaped = html.escape(note["content"][:200] + ("…" if len(note["content"]) > 200 else ""))
            ts_escaped = html.escape(note.get("timestamp", ""))
            st.markdown(
                f"""
                <div class="note-card">
                    <div class="note-source">{src_escaped}</div>
                    <div class="note-text">{text_escaped}</div>
                    <div class="note-ts">{ts_escaped}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("✕ Remove", key=f"del_note_{actual_idx}", use_container_width=True):
                st.session_state.notes.pop(actual_idx)
                st.rerun()

        st.divider()

        # Export notes
        export_text = "\n\n---\n\n".join(
            f"[{n.get('timestamp', '')}] {n.get('source', '')}\n\n{n['content']}"
            for n in notes
        )
        st.download_button(
            "⬇ Export Notes (.txt)",
            data=export_text,
            file_name="notebook_notes.txt",
            mime="text/plain",
            use_container_width=True,
            key="export_notes_btn",
        )

        if st.button("🗑 Clear all notes", key="clear_notes_btn", use_container_width=True):
            st.session_state.notes = []
            st.rerun()
    else:
        st.markdown(
            '<div style="color:#a8b3bc;font-size:.85rem;padding:.5rem 0;">No notes saved yet.<br>Save key insights from AI responses to build your research notes.</div>',
            unsafe_allow_html=True,
        )
