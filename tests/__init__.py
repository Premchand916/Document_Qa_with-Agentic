"""Offline regression baseline; run with unittest discovery from the repo root."""

import os

# Set before importing native libraries. This bounds this test process only.
os.environ.update({
    "OMP_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1", "HF_LOCAL_FILES_ONLY": "true",
    "GOOGLE_API_KEY": "", "GEMINI_API_KEY": "", "TAVILY_API_KEY": "",
    "LLM_PROVIDER": "ollama", "LANGSMITH_TRACING": "false",
    "LANGCHAIN_TRACING_V2": "false", "LANGSMITH_API_KEY": "",
    "PYTHON_DOTENV_DISABLED": "1",
    "DOCQA_AUTH_MODE": "local",
})
if os.environ.get("KMP_DUPLICATE_LIB_OK", "").lower() in {"true", "1", "yes"}:
    raise RuntimeError("Unset KMP_DUPLICATE_LIB_OK; the baseline must not use the unsafe runtime bypass")
