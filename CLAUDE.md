# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
# Install dependencies (first time)
pip install -r requirements.txt

# Start the FastAPI server (replaces Streamlit)
uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload --app-dir Document_Qa_with-Agentic

# Open in browser
open http://localhost:8000
```

`app/main.py` (Streamlit) is kept for reference but is no longer the primary entry point.

## Environment setup

Create a `.env` file in the repo root:

```
GOOGLE_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_key        # optional — needed for web search
HF_LOCAL_FILES_ONLY=true

# Optional Ollama overrides
LLM_PROVIDER=gemini                   # or "ollama"
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_FALLBACK_MODELS=llama3.2:1b,phi3:mini
```

## Architecture

The app is a Streamlit-based Document Intelligence system wired together with a LangGraph state machine.

### LangGraph workflow (`langgraph_flow/`)

`graph_builder.py` compiles the single graph used by the entire app. Every user query flows through `AgentState` (defined in `state.py`) as a typed dictionary. Two top-level paths exist:

- **Web search path**: `entry → web_search → qa_agent → END`
- **Document path**: `entry → intent_router → response_planner → planner → retriever → qa_agent → END`
  - The planner can branch to `dataframe_tool → END` for tabular queries instead of retriever.

### Agent layer (`agents/`)

| File | Role |
|---|---|
| `orchestrator_agent.py` | Factory for the active LLM (Gemini or Ollama). All other agents call `invoke_orchestrator()`. Handles Ollama memory-error fallback to smaller installed models automatically. |
| `intent_classifier.py` | Classifies query into one of 7 intents; falls back to keyword heuristics on LLM failure. |
| `response_planner.py` | Produces a `response_plan` dict (answer mode, user need, etc.). |
| `planner_agent.py` | Sets `use_tool=True` if the query targets tabular data, otherwise routes to retriever. |
| `retriever_agent.py` | Hybrid FAISS + BM25 retrieval fused via RRF (Reciprocal Rank Fusion), then cross-encoder reranked. Retrieval budget is intent-driven (e.g. `summarization` fetches k=8 from each, `document_qa` fetches k=5). |
| `qa_agent.py` | Final answer generation from retrieved context. |
| `web_search_agent.py` | Tavily API search; populates `documents` with web results. |
| `dataframe_tool.py` | Pandas-based computation for structured Excel/CSV queries. |

### Ingestion pipeline (`ingestion/`)

`file_loader.py` dispatches by extension to format-specific loaders:
- PDF → `pdfplumber`, page-by-page
- PPTX → `python-pptx`, slide-by-slide
- Excel/CSV/TSV → `pandas`, chunked into 20-row `Document` objects plus a summary header
- TXT/MD/JSON → raw text decode
- Images (JPG/PNG/TIFF/BMP/WebP) → OCR via `pytesseract` (requires system Tesseract) or Gemini Vision fallback when `GOOGLE_API_KEY` is set

`semantic_chunker.py` post-processes raw documents into overlapping semantic chunks suitable for FAISS indexing.

`extract_tabular_assets()` in `file_loader.py` runs in parallel with `load_uploaded_file()` and stores live `pandas.DataFrame` objects in `AgentState.tabular_assets` for direct computation (bypassing the vector store).

### Vector store (`vector_Store/`)

FAISS index is persisted to `vector_db/` at the repo root via `create_vector_store()` and reloaded on app startup via `load_vector_store()`. Embeddings use `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, loaded once and cached via `lru_cache`).

### Web UI (`app/server.py` + `app/static/index.html`)

FastAPI server exposes REST endpoints (`POST /api/upload`, `POST /api/chat`, `GET /api/state`, `POST /api/provider`, `GET /api/prompt-library`) and serves the single-page app from `app/static/`. The SPA is built with Alpine.js (reactive state) + marked.js (markdown rendering) — no build step required. Session state is held in a module-level `_AppState` singleton; `clear_retrieval_caches()` is called on every new upload to invalidate the BM25 and FAISS LRU caches.

### Prompt utilities (`utils/`, `app/prompt_library.py`)

`prompt_skill.py` loads an optional YAML prompt-profile file at `prompt_skill_path`, which wraps the user query in custom instructions before it reaches the graph. `prompt_library.py` defines categorised prompt templates shown in the UI.

## Key env-var controlled behaviours

- `LLM_PROVIDER` (`gemini` | `ollama`) — toggled live from the sidebar; written back to `os.environ` immediately.
- `OLLAMA_ACTIVE_MODEL` — set automatically to whichever Ollama model is actually responding (may differ from `OLLAMA_MODEL` after fallback).
- `HF_LOCAL_FILES_ONLY=true` — prevents HuggingFace from hitting the network on every startup once the embedding model is cached locally.
