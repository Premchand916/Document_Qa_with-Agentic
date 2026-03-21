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

FILE_TYPE_COLORS = {
    "pdf": "#e05252", "pptx": "#e07c52", "xlsx": "#52a852", "xlsm": "#52a852",
    "csv": "#52a87c", "tsv": "#52a8a8", "txt": "#a8a852", "md": "#a852a8", "json": "#5288a8",
}
FILE_TYPE_LABELS = {
    "pdf": "PDF", "pptx": "PPTX", "xlsx": "XLSX", "xlsm": "XLSX",
    "csv": "CSV", "tsv": "TSV", "txt": "TXT", "md": "MD", "json": "JSON",
}

SOURCE_STOPWORDS = {
    "about", "after", "again", "being", "below", "between", "could", "from",
    "have", "into", "just", "more", "most", "other", "same", "some", "such",
    "than", "that", "them", "then", "there", "these", "they", "this", "very",
    "what", "when", "where", "which", "while", "with", "would", "your",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    for document in documents or []:
        metadata = getattr(document, "metadata", {}) or {}
        source_name = metadata.get("source", "Unknown document")
        page = metadata.get("page", "N/A")
        file_type = str(metadata.get("file_type", "")).lower().lstrip(".")
        excerpt = clean_text(getattr(document, "page_content", ""))
        if not excerpt:
            continue
        dedupe_key = (source_name, page, excerpt[:160])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        location = format_location_label(page)
        title = source_name if not location else f"{source_name} - {location}"
        raw_score = metadata.get("rerank_score")
        if raw_score is not None:
            score_pct = min(100, max(0, int((float(raw_score) + 10) * 5)))
        else:
            score_pct = None
        cards.append({
            "title": title,
            "summary": summarize_words(excerpt, max_words=20),
            "excerpt": excerpt,
            "file_type": file_type,
            "file_type_label": FILE_TYPE_LABELS.get(file_type, file_type.upper() or "DOC"),
            "file_type_color": FILE_TYPE_COLORS.get(file_type, "#a8b3bc"),
            "relevance_score": score_pct,
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


def build_format_pills(file_types):
    if not file_types:
        return '<span class="pill">Formats appear after upload</span>'
    return "".join(f'<span class="pill">{html.escape(ft)}</span>' for ft in file_types)


def _file_badge_html(label, color):
    return (
        f'<span class="file-badge" style="background:{color}22;'
        f'color:{color};border:1px solid {color}44;">{html.escape(label)}</span>'
    )


def _relevance_bar_html(score_pct):
    if score_pct is None:
        return ""
    if score_pct >= 70:
        bar_color, quality = "#52a87c", "High relevance"
    elif score_pct >= 40:
        bar_color, quality = "#f6b44c", "Moderate relevance"
    else:
        bar_color, quality = "#e05252", "Low relevance"
    return (
        f'<div class="relevance-bar-wrap">'
        f'<div class="relevance-bar-bg">'
        f'<div class="relevance-bar-fill" style="width:{score_pct}%;background:{bar_color};"></div>'
        f'</div>'
        f'<span class="relevance-label" style="color:{bar_color};">{quality}</span>'
        f'</div>'
    )


# ── Styles ────────────────────────────────────────────────────────────────────

def render_styles():
    st.markdown(
        """
        <style>
        :root {
            --bg-1:#081118; --bg-2:#101a21; --line:rgba(255,255,255,.08);
            --text:#f6f7f9; --muted:#a8b3bc; --accent:#f6b44c; --teal:#5dc8ba;
        }
        .stApp {
            background: radial-gradient(circle at top left, rgba(93,200,186,.16), transparent 26%),
                radial-gradient(circle at top right, rgba(246,180,76,.14), transparent 28%),
                linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
            color: var(--text);
        }
        .main .block-container { max-width:1180px; padding-top:2rem; padding-bottom:3rem; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(10,17,23,.95), rgba(15,23,30,.96));
            border-right: 1px solid var(--line);
        }
        .hero-shell, .section-shell {
            border:1px solid var(--line); border-radius:24px;
            background: linear-gradient(180deg, rgba(15,23,30,.9), rgba(10,17,23,.92));
        }
        .hero-shell { padding:1.8rem; margin-bottom:1rem; box-shadow:0 24px 70px rgba(0,0,0,.24); }
        .section-shell { padding:1.1rem 1.2rem .25rem 1.2rem; margin-top:.7rem; }
        .eyebrow { margin:0 0 .65rem 0; color:var(--accent); font-size:.8rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
        .hero-title { margin:0; color:var(--text); font-size:clamp(2rem,5vw,3.05rem); line-height:1.04; font-weight:700; }
        .hero-copy, .section-copy, .library-meta, .mini-note { color:var(--muted); line-height:1.6; }
        .hero-copy { margin:.9rem 0 0 0; max-width:780px; font-size:1.02rem; }
        .section-title { margin:0; color:var(--text); font-size:1.08rem; font-weight:700; }
        .section-copy { margin:.4rem 0 1rem 0; }
        .status-row, .pill-row { display:flex; flex-wrap:wrap; gap:.75rem; margin-top:1rem; }
        .status-card { min-width:185px; padding:.85rem 1rem; border-radius:18px; border:1px solid var(--line); background:rgba(255,255,255,.035); }
        .status-label, .source-label { display:block; color:var(--muted); font-size:.78rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
        .status-value { display:block; margin-top:.35rem; color:var(--text); font-size:1rem; font-weight:600; }
        .pill { display:inline-flex; align-items:center; padding:.42rem .7rem; border-radius:999px; border:1px solid var(--line); background:rgba(255,255,255,.04); color:var(--text); font-size:.82rem; font-weight:600; }
        .sidebar-note, .empty-shell, .error-shell, .source-detail-shell {
            padding:1rem 1.05rem; border-radius:18px; border:1px solid var(--line);
            background:rgba(255,255,255,.03); color:var(--muted); line-height:1.55;
        }
        .empty-shell { border-style:dashed; background:rgba(255,255,255,.025); margin:.4rem 0 1rem 0; }
        .error-shell { border-color:rgba(255,126,126,.24); background:rgba(111,32,32,.18); margin-top:1rem; }
        .error-label { color:#ffd6d6; font-weight:700; }
        .source-label { margin:1rem 0 .55rem 0; letter-spacing:.12em; }
        .source-summary { margin:-.3rem 0 .75rem .1rem; color:var(--muted); font-size:.98rem; line-height:1.55; }
        .source-detail-meta { color:var(--accent); font-size:.82rem; letter-spacing:.08em; text-transform:uppercase; }
        .source-detail-copy { margin-top:.7rem; color:var(--text); line-height:1.7; }
        .source-detail-copy mark { background:rgba(246,180,76,.28); color:var(--text); padding:.02rem .12rem; border-radius:.2rem; }
        div.stButton > button {
            width:100%; min-height:4.2rem; padding:1rem 1.05rem; margin-bottom:.72rem;
            border-radius:18px; border:1px solid var(--line);
            background: linear-gradient(135deg, rgba(246,180,76,.08), rgba(93,200,186,.06)),
                linear-gradient(180deg, rgba(18,28,36,.98), rgba(12,20,27,.98));
            color:var(--text); font-size:.98rem; font-weight:600; line-height:1.4;
            text-align:left; box-shadow:0 14px 35px rgba(0,0,0,.16);
            transition:all .18s ease; cursor:pointer;
        }
        div.stButton > button:hover {
            border-color:rgba(246,180,76,.35);
            box-shadow:0 0 0 1px rgba(246,180,76,.18), 0 14px 35px rgba(0,0,0,.22);
            background: linear-gradient(135deg, rgba(246,180,76,.12), rgba(93,200,186,.08)),
                linear-gradient(180deg, rgba(18,28,36,.98), rgba(12,20,27,.98));
        }
        [data-testid="stChatMessage"] { border:1px solid var(--line); border-radius:22px; padding:.45rem .8rem; background:rgba(255,255,255,.03); }
        [data-testid="stChatInput"] { background:rgba(10,17,23,.94); border:1px solid var(--line); border-radius:20px; transition:all .18s; }
        [data-testid="stChatInput"]:focus-within { border-color:rgba(246,180,76,.4); box-shadow:0 0 0 2px rgba(246,180,76,.1); }
        [data-testid="stChatInput"] textarea { color:var(--text); }
        ::-webkit-scrollbar { width:6px; }
        ::-webkit-scrollbar-track { background:var(--bg-1); }
        ::-webkit-scrollbar-thumb { background:rgba(255,255,255,.12); border-radius:3px; }
        ::-webkit-scrollbar-thumb:hover { background:rgba(246,180,76,.3); }
        .stTabs [data-baseweb="tab-list"] {
            gap:4px; background:rgba(255,255,255,.03); border-radius:14px;
            padding:4px; border:1px solid var(--line);
        }
        .stTabs [data-baseweb="tab"] { border-radius:10px; color:var(--muted); font-size:.85rem; font-weight:600; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background:rgba(246,180,76,.14); color:var(--accent); }
        .stTabs [data-baseweb="tab-panel"] { padding-top:1rem; }
        [data-testid="stDownloadButton"] > button {
            background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.12);
            color:var(--muted); border-radius:8px; padding:4px 14px;
            font-size:.78rem; min-height:2rem; transition:all .18s;
        }
        [data-testid="stDownloadButton"] > button:hover { border-color:rgba(246,180,76,.4); color:var(--text); }
        .action-btn {
            background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.1);
            color:var(--muted); border-radius:8px; padding:4px 12px; cursor:pointer;
            font-size:.78rem; font-family:inherit; transition:all .18s;
        }
        .action-btn:hover { border-color:rgba(246,180,76,.4); color:var(--text); }
        .breadcrumb-chip { display:inline-block; border-radius:6px; padding:2px 10px; font-size:.74rem; font-weight:700; }
        .file-badge { display:inline-block; border-radius:6px; padding:1px 7px; font-size:.7rem; font-weight:700; margin-right:4px; }
        .relevance-bar-wrap { display:flex; align-items:center; gap:8px; margin:.25rem 0; }
        .relevance-bar-bg { flex:1; height:4px; background:rgba(255,255,255,.08); border-radius:2px; }
        .relevance-bar-fill { height:100%; border-radius:2px; }
        .relevance-label { font-size:.72rem; font-weight:700; white-space:nowrap; }
        .file-row { display:flex; align-items:center; gap:8px; padding:5px 0; font-size:.82rem; border-bottom:1px solid rgba(255,255,255,.04); }
        .file-row-name { color:#f6f7f9; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .file-row-size { color:#a8b3bc; flex-shrink:0; }
        .rating-confirmed { color:#52c878; font-size:.78rem; font-weight:600; }
        .rating-negative-confirmed { color:#e05252; font-size:.78rem; font-weight:600; }
        @media (max-width:640px) {
            .main .block-container { padding-top:1.3rem; }
            .hero-shell { padding:1.2rem; }
            .section-shell { padding:1rem 1rem .1rem 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────

def render_header():
    has_vectorstore = st.session_state.vectorstore is not None
    document_status = (
        f"{st.session_state.document_count} chunks indexed"
        if has_vectorstore and st.session_state.document_count
        else "Knowledge base ready" if has_vectorstore else "Waiting for upload"
    )
    file_type_status = (
        ", ".join(st.session_state.uploaded_file_types)
        if st.session_state.uploaded_file_types
        else "Upload files to see active formats"
    )
    question_status = (
        f"{len(st.session_state.suggested_questions)} document prompts"
        if st.session_state.suggested_questions
        else "Generated after upload"
    )
    answer_count = len([m for m in st.session_state.chat_history if m["role"] == "assistant"])
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


# ── Error panel ───────────────────────────────────────────────────────────────

def render_error_panel():
    if not st.session_state.last_error:
        return
    last_error = st.session_state.last_error
    st.markdown(
        f"""
        <div class="error-shell">
            <p class="error-label">Issue in {last_error["stage"]}</p>
            <p>{html.escape(last_error["message"])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Technical details", expanded=False):
        st.code(last_error["details"] or last_error["message"])


# ── Source cards ──────────────────────────────────────────────────────────────

def render_source_cards(message_index, message):
    source_cards = message.get("source_cards") or []
    if not source_cards:
        return
    st.markdown(
        f'<div class="source-label">Sources Referenced — {len(source_cards)}</div>',
        unsafe_allow_html=True,
    )
    for source_index, card in enumerate(source_cards):
        card_key = f"source_{message_index}_{source_index}"
        badge = _file_badge_html(card.get("file_type_label", "DOC"), card.get("file_type_color", "#a8b3bc"))
        st.markdown(badge, unsafe_allow_html=True)
        if st.button(card["title"], key=f"source_button_{message_index}_{source_index}", use_container_width=True):
            toggle_source_card(card_key)
        rel_html = _relevance_bar_html(card.get("relevance_score"))
        if rel_html:
            st.markdown(rel_html, unsafe_allow_html=True)
        st.markdown(f'<div class="source-summary">{html.escape(card["summary"])}</div>', unsafe_allow_html=True)
        if st.session_state.selected_source_key == card_key:
            highlighted = highlight_excerpt(card["excerpt"], message.get("query", ""))
            st.markdown(
                f"""
                <div class="source-detail-shell">
                    <div class="source-detail-meta">{html.escape(card["title"])}</div>
                    <div class="source-detail-copy">{highlighted}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Answer breadcrumb ─────────────────────────────────────────────────────────

def _render_answer_breadcrumb(response_meta):
    if not response_meta:
        return
    intent = response_meta.get("intent", "document_qa").replace("_", " ").title()
    mode = response_meta.get("answer_mode", "Auto")
    user_need = response_meta.get("user_need", "")
    sections = response_meta.get("sections", [])
    sections_str = " / ".join(sections[:3]) if sections else ""

    chips = (
        f'<span class="breadcrumb-chip" style="background:rgba(246,180,76,.12);color:#f6b44c;">'
        f'{html.escape(intent)}</span>'
        f'<span style="color:#a8b3bc;font-size:.74rem;">&#8594;</span>'
        f'<span class="breadcrumb-chip" style="background:rgba(93,200,186,.1);color:#5dc8ba;margin-left:4px;">'
        f'{html.escape(mode)}</span>'
    )
    if sections_str:
        chips += (
            f'<span style="color:#a8b3bc;font-size:.74rem;margin-left:4px;">&#8594;</span>'
            f'<span style="color:#a8b3bc;font-size:.74rem;margin-left:6px;">{html.escape(sections_str)}</span>'
        )
    breadcrumb = (
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:4px;margin:.4rem 0 .3rem 0;">'
        f'{chips}</div>'
    )
    if user_need:
        breadcrumb += (
            f'<div style="color:#a8b3bc;font-size:.8rem;margin-bottom:.4rem;">'
            f'{html.escape(user_need)}</div>'
        )
    st.markdown(breadcrumb, unsafe_allow_html=True)


# ── Chat message ──────────────────────────────────────────────────────────────

def render_chat_message(message, message_index):
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant":
            response_meta = message.get("response_meta", {}) or {}
            _render_answer_breadcrumb(response_meta)

            # Action row: Copy | Save | Helpful | Not helpful
            a1, a2, a3, a4, _ = st.columns([1.1, 1.1, 1.3, 1.6, 4])
            with a1:
                escaped_content = html.escape(message["content"])
                copy_id = f"copy_content_{message_index}"
                st.markdown(
                    f'<span id="{copy_id}" style="display:none">{escaped_content}</span>'
                    f'<button class="action-btn" '
                    f'onclick="navigator.clipboard.writeText('
                    f'document.getElementById(\'{copy_id}\').textContent)">'
                    f'Copy</button>',
                    unsafe_allow_html=True,
                )
            with a2:
                st.download_button(
                    label="Save",
                    data=message["content"],
                    file_name=f"answer_{message_index}.md",
                    mime="text/markdown",
                    key=f"download_{message_index}",
                )
            existing_rating = st.session_state.answer_ratings.get(message_index, {})
            with a3:
                if existing_rating.get("rating") == 1:
                    st.markdown('<span class="rating-confirmed">Marked helpful</span>', unsafe_allow_html=True)
                else:
                    if st.button("Helpful", key=f"rate_up_{message_index}"):
                        st.session_state.answer_ratings[message_index] = {"rating": 1}
                        st.rerun()
            with a4:
                if existing_rating.get("rating") == -1:
                    st.markdown('<span class="rating-negative-confirmed">Marked unhelpful</span>', unsafe_allow_html=True)
                else:
                    if st.button("Not helpful", key=f"rate_down_{message_index}"):
                        st.session_state.answer_ratings[message_index] = {"rating": -1}
                        st.rerun()

            # Regenerate
            if message.get("query"):
                regen_col, _ = st.columns([2, 8])
                with regen_col:
                    if st.button("Regenerate", key=f"regen_{message_index}"):
                        queue_prompt(message["query"])

            render_source_cards(message_index, message)


# ── Prompt library ────────────────────────────────────────────────────────────

def render_prompt_library():
    st.markdown(
        """
        <section class="section-shell">
            <h2 class="section-title">Prompt Library</h2>
            <p class="section-copy">72 curated prompts across 12 categories. Click any to run it instantly.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    categories = get_prompt_categories()
    tab_labels = [cat.split(" And ")[0][:13] for cat in categories]
    tabs = st.tabs(tab_labels)
    for tab, category_key in zip(tabs, categories):
        with tab:
            category_data = PROMPT_LIBRARY[category_key]
            st.caption(category_data["description"])
            left_col, right_col = st.columns(2)
            for index, prompt in enumerate(category_data["prompts"]):
                target = left_col if index % 2 == 0 else right_col
                with target:
                    if st.button(prompt, key=f"library_prompt_{category_key}_{index}", use_container_width=True):
                        queue_prompt(prompt)


# ── Document prompts ──────────────────────────────────────────────────────────

def render_document_prompts():
    if st.session_state.suggested_questions:
        left_col, right_col = st.columns(2)
        for index, question in enumerate(st.session_state.suggested_questions):
            target = left_col if index % 2 == 0 else right_col
            with target:
                if st.button(question, key=f"suggested_question_{index}", use_container_width=True):
                    queue_prompt(question)
    else:
        st.markdown(
            '<div class="empty-shell">Upload supported files to generate tailored prompts from your content.</div>',
            unsafe_allow_html=True,
        )


# ── Answer configuration (interactive) ───────────────────────────────────────

def render_answer_blueprint():
    st.markdown(
        """
        <section class="section-shell">
            <h2 class="section-title">Answer Configuration</h2>
            <p class="section-copy">Set answer mode, audience, and depth. Changes apply to the next question.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    col_mode, col_audience, col_depth = st.columns(3)
    with col_mode:
        st.selectbox("Answer Mode", ANSWER_MODES, key="answer_mode")
        st.caption(MODE_DESCRIPTIONS[st.session_state.answer_mode])
    with col_audience:
        st.selectbox("Audience", AUDIENCE_OPTIONS, key="audience")
        st.caption(AUDIENCE_DESCRIPTIONS[st.session_state.audience])
    with col_depth:
        st.select_slider("Depth", options=DEPTH_OPTIONS, key="response_depth")
        st.caption(DEPTH_DESCRIPTIONS[st.session_state.response_depth])


# ── Conversation export ───────────────────────────────────────────────────────

def render_conversation_export():
    if not st.session_state.chat_history:
        return
    lines = ["# Document QA Conversation\n"]
    for msg in st.session_state.chat_history:
        role_label = "Question" if msg["role"] == "user" else "Answer"
        lines.append(f"## {role_label}")
        lines.append(msg["content"])
        if msg["role"] == "assistant" and msg.get("source_cards"):
            lines.append("\n**Sources:**")
            for card in msg["source_cards"]:
                lines.append(f"- {card['title']}")
        lines.append("")
    export_md = "\n".join(lines)
    plain = re.sub(r"#+\s*", "", export_md).replace("**", "")

    exp1, exp2, _ = st.columns([1.3, 1.3, 7])
    with exp1:
        st.download_button(
            "Export MD",
            data=export_md,
            file_name="conversation.md",
            mime="text/markdown",
            key="export_md_button",
        )
    with exp2:
        st.download_button(
            "Export TXT",
            data=plain,
            file_name="conversation.txt",
            mime="text/plain",
            key="export_txt_button",
        )


# ── Prepare documents ─────────────────────────────────────────────────────────

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


# ── State initialization ──────────────────────────────────────────────────────

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
        "query_history": [],
        "answer_ratings": {},
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


# ── Bootstrap ─────────────────────────────────────────────────────────────────

initialize_state()
render_styles()

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("## Knowledge Base")
st.sidebar.markdown(
    """
    <div class="sidebar-note">
        Upload mixed file types to build a searchable knowledge base. The AI plans,
        retrieves, and synthesizes answers across all your documents.
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("### Supported Formats")
st.sidebar.caption(", ".join(sorted(label for label in SUPPORTED_FILE_TYPES.values())))

uploaded_files = st.sidebar.file_uploader(
    "Upload business files",
    type=[ext.lstrip(".") for ext in SUPPORTED_FILE_TYPES.keys()],
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
            st.sidebar.success(
                f"Indexed {len(chunked_docs)} chunks from {source_count} "
                f"files across {', '.join(file_types)}."
            )

if st.session_state.vectorstore is not None:
    st.sidebar.success("Knowledge base is ready.")
if st.session_state.uploaded_file_types:
    st.sidebar.caption("Active formats: " + ", ".join(st.session_state.uploaded_file_types))

# File management panel
if uploaded_files and st.session_state.vectorstore is not None:
    st.sidebar.markdown("### Loaded Files")
    for ufile in uploaded_files:
        ext = Path(ufile.name).suffix.lstrip(".").lower()
        label = FILE_TYPE_LABELS.get(ext, ext.upper())
        color = FILE_TYPE_COLORS.get(ext, "#a8b3bc")
        size_kb = round(getattr(ufile, "size", 0) / 1024, 1)
        st.sidebar.markdown(
            f'<div class="file-row">'
            f'<span class="file-badge" style="background:{color}22;color:{color};'
            f'border:1px solid {color}44;">{label}</span>'
            f'<span class="file-row-name">{html.escape(ufile.name)}</span>'
            f'<span class="file-row-size">{size_kb}KB</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# Sticky document prompts in sidebar
if st.session_state.suggested_questions:
    st.sidebar.markdown("### Document Prompts")
    for index, question in enumerate(st.session_state.suggested_questions):
        label = question[:55] + ("..." if len(question) > 55 else "")
        if st.sidebar.button(label, key=f"sidebar_suggested_{index}", help=question, use_container_width=True):
            queue_prompt(question)

# ── Main content ──────────────────────────────────────────────────────────────

render_header()

graph = None
try:
    graph = get_graph()
except Exception as exc:
    record_error("building the app workflow", exc)
render_error_panel()

tab_chat, tab_prompts, tab_config = st.tabs(["Chat", "Prompts", "Config"])

with tab_config:
    render_answer_blueprint()

with tab_prompts:
    render_prompt_library()
    st.markdown("---")
    with st.expander(
        "Document-Specific Prompts",
        expanded=bool(st.session_state.suggested_questions),
    ):
        st.markdown(
            '<p style="color:var(--muted);font-size:.9rem;margin-bottom:.5rem;">'
            'Generated from your uploaded files and balanced across sources.</p>',
            unsafe_allow_html=True,
        )
        render_document_prompts()

with tab_chat:
    render_conversation_export()

    for message_index, message in enumerate(st.session_state.chat_history):
        render_chat_message(message, message_index)

    if not st.session_state.chat_history:
        st.markdown(
            '<div class="empty-shell" style="margin-top:1rem;">'
            'No conversation yet. Ask a question below, or pick a prompt from the Prompts tab.'
            '</div>',
            unsafe_allow_html=True,
        )

# Query history dropdown (above chat input)
if st.session_state.query_history:
    hist_col, _ = st.columns([4, 6])
    with hist_col:
        selected_history = st.selectbox(
            "Recent queries",
            ["— Recent queries —"] + st.session_state.query_history,
            key="history_selector",
            label_visibility="collapsed",
        )
        if (
            selected_history
            and selected_history != "— Recent queries —"
            and selected_history != st.session_state.get("_last_history_select")
        ):
            st.session_state["_last_history_select"] = selected_history
            queue_prompt(selected_history)

typed_query = st.chat_input(
    "Ask for summaries, comparisons, metrics, risks, requirements, action plans, or cross-file insights",
    key="chat_input",
)
query = st.session_state.pending_query or typed_query
if st.session_state.pending_query:
    st.session_state.pending_query = None

if query:
    # Save to query history (max 20, no duplicates)
    if query not in st.session_state.query_history:
        st.session_state.query_history.insert(0, query)
        st.session_state.query_history = st.session_state.query_history[:20]

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
            "response_meta": {
                "answer_mode": st.session_state.answer_mode,
                "intent": "document_qa",
                "user_need": "Upload files first.",
            },
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
            "response_meta": {
                "answer_mode": st.session_state.answer_mode,
                "intent": "document_qa",
                "user_need": "Resolve the workflow issue.",
            },
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
            with st.status("Processing your question...", expanded=True) as status:
                state["step_tracker"] = {"status": status}
                result = graph.invoke(state)
                status.update(label="Answer ready", state="complete", expanded=False)
        except Exception as exc:
            record_error("running the query", exc)
            response = "I hit an error while running that question. Check the issue panel above for the cause."
            assistant_message = {
                "role": "assistant",
                "content": response,
                "query": query,
                "source_cards": [],
                "response_meta": {
                    "answer_mode": st.session_state.answer_mode,
                    "intent": "document_qa",
                    "user_need": "Recover from the execution error.",
                },
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
