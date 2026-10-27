# Repository Guidelines

## Project Structure & Module Organization

`app/server.py` serves the FastAPI API and the single-page UI in `app/static/index.html`; `app/main.py` is an older Streamlit entry point. Document parsing and chunking live in `ingestion/`, embeddings in `embeddings/`, FAISS construction in `vector_Store/`, workspace persistence in `storage/`, and hybrid retrieval in `retrieval/` and `agents/retriever_agent.py`. `langgraph_flow/` defines workflow state and routing; `agents/` implements the workflow steps. Shared helpers are in `utils/`, `tools/`, and `memory/`. Tests: `tests/`. Documentation images: `docs/images/`.

## Build, Test, and Development Commands

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
DOCQA_AUTH_MODE=local python -m uvicorn app.server:app --reload --host 127.0.0.1 --port 8000 --no-proxy-headers
```

Open `http://127.0.0.1:8000`. Run `python -m compileall -q agents app embeddings ingestion langgraph_flow retrieval tools utils vector_Store storage memory` for syntax checks.

Install test dependencies with `python -m pip install -r requirements-dev.txt`. Run `python -m tools.run_baseline` for offline tests, failure diagnostics, and an environment snapshot; results go to ignored `.baseline-results/`.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions and modules, `PascalCase` classes, and uppercase constants. Add type hints to new public interfaces; keep route handlers thin. Preserve the existing `vector_Store` package name. No formatter or linter is configured; avoid unrelated formatting changes.

## Testing Guidelines

Tests use standard-library `unittest`; run `python -m unittest discover -s tests -t . -v`. Add focused desired-behavior tests in `tests/test_*.py`, with synthetic inputs and temporary stores. Application tests block network/model loading. Keep failure diagnostics separate from regression tests. No coverage threshold is configured. See `docs/BASELINE.md` for evaluation; UI changes need browser checks.

## Commit & Pull Request Guidelines

Recent commits use concise imperative subjects, sometimes prefixed `feat:` or `fix:`; use those prefixes for feature and bug changes. Pull requests should explain the behavior changed, list verification performed, link related issues, and include screenshots for UI changes. Open an issue first for major changes, as requested in `README.md`.

## Security & Configuration

Store settings in ignored `.env`. Set `GOOGLE_API_KEY` for Gemini or use local Ollama; `TAVILY_API_KEY` is needed for web search. Never commit keys, uploaded documents, or generated `vector_db/` indexes.

Access is disabled unless configured. Use explicit loopback-only `DOCQA_AUTH_MODE=local` for development; see `docs/STEP1_RESULTS.md` for the authentication-proxy boundary. See `docs/STEP2_RESULTS.md` for workspace memberships. Publish through `WorkspaceRepository`; never load legacy pickles or mutate provider environment settings per request.
