# Repository Guidelines

## Project Structure & Module Organization

`app/server.py` serves the FastAPI API and the single-page UI in `app/static/index.html`; `app/main.py` is an older Streamlit entry point. Document parsing and chunking live in `ingestion/`, embeddings in `embeddings/`, FAISS storage in `vector_Store/`, and hybrid retrieval in `retrieval/` and `agents/retriever_agent.py`. `langgraph_flow/` defines workflow state and routing; `agents/` implements the workflow steps. Shared helpers are in `utils/`, `tools/`, and `memory/`. Keep new tests in `tests/` and documentation images in `docs/images/`.

## Build, Test, and Development Commands

Run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app.server:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The UI has no separate build step. Run `python -m compileall -q agents app embeddings ingestion langgraph_flow retrieval tools utils vector_Store memory` for a quick syntax check.

## Coding Style & Naming Conventions

Follow nearby Python code: four-space indentation, `snake_case` functions and modules, `PascalCase` classes, and uppercase constants. Add type hints to new public interfaces and keep route handlers thin by placing document and workflow logic in their existing packages. Preserve the existing `vector_Store` package name. No formatter or linter is configured; avoid unrelated formatting changes.

## Testing Guidelines

No automated test suite or coverage threshold is currently configured. For new behavior, add focused `tests/test_*.py` tests and declare a test runner when introducing the first tests. Until then, run the syntax check and smoke-test upload, document chat, provider selection, and relevant error responses in the FastAPI UI. Note any model, API key, or network dependency in the test instructions.

## Commit & Pull Request Guidelines

Recent commits use concise imperative subjects, sometimes prefixed `feat:` or `fix:`; use those prefixes for feature and bug changes. Keep each commit focused. Pull requests should explain the behavior changed, list verification performed, link related issues, and include screenshots for UI changes. Open an issue first for major changes, as requested in `README.md`.

## Security & Configuration

Store local settings in `.env` (ignored by Git). Set `GOOGLE_API_KEY` for Gemini or use local Ollama; `TAVILY_API_KEY` is needed for web search. Never commit keys, uploaded documents, or generated `vector_db/` indexes.
