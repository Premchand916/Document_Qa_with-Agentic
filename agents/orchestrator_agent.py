import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


# ── Gemini (Google) ───────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_gemini_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0,
    )


# ── Ollama (local) ────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_ollama_llm():
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        try:
            from langchain_community.chat_models import ChatOllama
        except ImportError as exc:
            raise ImportError(
                "Ollama LangChain package not found. "
                "Install it with: pip install langchain-ollama"
            ) from exc

    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )


# ── Provider selector ─────────────────────────────────────────────────────────
def create_orchestrator():
    """Return the active LLM based on LLM_PROVIDER env var.

    Supported values:
        gemini  (default) — Google Gemini via GOOGLE_API_KEY
        ollama            — Local Ollama server (no API key needed)
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
    if provider == "ollama":
        return _get_ollama_llm()
    return _get_gemini_llm()
