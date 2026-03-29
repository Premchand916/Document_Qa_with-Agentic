import json
import os
from functools import lru_cache
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

DEFAULT_OLLAMA_FALLBACK_MODELS = (
    "llama3.2:3b",
    "llama3.2:1b",
    "phi3:mini",
    "gemma2:2b",
    "qwen2.5:1.5b",
    "qwen2.5:0.5b",
)
SMALL_MODEL_HINTS = ("0.5b", "1b", "1.5b", "2b", "3b", "mini", "small")


class LLMConfigurationError(RuntimeError):
    """Raised when the active LLM provider is unavailable or misconfigured."""


class OllamaModelMemoryError(RuntimeError):
    """Raised when the configured Ollama model does not fit in memory."""


@lru_cache(maxsize=1)
def _get_gemini_llm():
    if not os.getenv("GOOGLE_API_KEY"):
        raise LLMConfigurationError(
            "Gemini is selected but `GOOGLE_API_KEY` is not configured. "
            "Add it to your `.env` file or switch the app to Ollama."
        )

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0,
    )


def _get_ollama_settings():
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2").strip() or "llama3.2"
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip() or "http://localhost:11434"
    return model_name, base_url.rstrip("/")


@lru_cache(maxsize=12)
def _get_ollama_llm(model_name, base_url):
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
        model=model_name,
        base_url=base_url,
        temperature=0,
    )


def _is_ollama_memory_error(exc):
    message = str(exc).lower()
    return "requires more system memory" in message or (
        "system memory" in message and "status code: 500" in message
    )


def _is_ollama_unavailable_error(exc):
    message = str(exc).lower()
    needles = (
        "connection refused",
        "failed to connect",
        "could not connect",
        "max retries exceeded",
        "winerror 10061",
        "winerror 10013",
        "actively refused",
        "connect error",
    )
    return any(needle in message for needle in needles)


def _build_ollama_unavailable_message(base_url):
    return (
        f"Ollama is selected but the server is unavailable at {base_url}. "
        "Start Ollama and make sure the API is listening there, or switch the app to Gemini."
    )


@lru_cache(maxsize=4)
def _get_installed_ollama_models(base_url):
    request = Request(f"{base_url}/api/tags", headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, ValueError):
        return tuple()

    models = []
    for item in payload.get("models", []):
        model_name = str(item.get("name") or "").strip()
        if model_name and model_name not in models:
            models.append(model_name)
    return tuple(models)


def _resolve_installed_model(candidate, installed_models):
    candidate_lower = candidate.lower()
    for installed_model in installed_models:
        installed_lower = installed_model.lower()
        if installed_lower == candidate_lower or installed_lower.startswith(f"{candidate_lower}:"):
            return installed_model
    return ""


def _get_fallback_candidates(primary_model, base_url):
    configured_fallbacks = [
        item.strip()
        for item in os.getenv("OLLAMA_FALLBACK_MODELS", "").split(",")
        if item.strip()
    ]
    seed_models = configured_fallbacks or list(DEFAULT_OLLAMA_FALLBACK_MODELS)
    installed_models = list(_get_installed_ollama_models(base_url))
    candidates = []

    def add_candidate(model_name):
        if not model_name:
            return
        if model_name.lower() == primary_model.lower():
            return
        if model_name not in candidates:
            candidates.append(model_name)

    if installed_models:
        for seed_model in seed_models:
            add_candidate(_resolve_installed_model(seed_model, installed_models) or seed_model)

        for installed_model in installed_models:
            if any(hint in installed_model.lower() for hint in SMALL_MODEL_HINTS):
                add_candidate(installed_model)
    else:
        for seed_model in seed_models:
            add_candidate(seed_model)

    return candidates


def _build_memory_error_message(primary_model, attempted_models):
    attempted_text = ""
    if attempted_models:
        attempted_text = (
            " I also tried smaller local models "
            f"({', '.join(attempted_models[:4])}), but none could be used automatically."
        )

    return (
        f'Ollama model "{primary_model}" needs more RAM than is currently available on this machine.'
        f"{attempted_text} "
        "Set `OLLAMA_MODEL` to a lighter model such as `llama3.2:1b` or `phi3:mini`, "
        "or switch the app back to Gemini. If you do not have a small model installed yet, "
        "run `ollama pull llama3.2:1b` first."
    )


def create_orchestrator():
    """Return the active LLM based on LLM_PROVIDER env var."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
    if provider == "ollama":
        model_name, base_url = _get_ollama_settings()
        os.environ["OLLAMA_ACTIVE_MODEL"] = model_name
        return _get_ollama_llm(model_name, base_url)
    return _get_gemini_llm()


def invoke_orchestrator(prompt):
    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
    if provider != "ollama":
        return _get_gemini_llm().invoke(prompt)

    primary_model, base_url = _get_ollama_settings()
    os.environ["OLLAMA_ACTIVE_MODEL"] = primary_model

    try:
        return _get_ollama_llm(primary_model, base_url).invoke(prompt)
    except Exception as exc:
        if _is_ollama_unavailable_error(exc):
            raise LLMConfigurationError(_build_ollama_unavailable_message(base_url)) from exc
        if not _is_ollama_memory_error(exc):
            raise

    attempted_models = []
    for candidate_model in _get_fallback_candidates(primary_model, base_url):
        attempted_models.append(candidate_model)
        try:
            response = _get_ollama_llm(candidate_model, base_url).invoke(prompt)
        except Exception as fallback_exc:
            message = str(fallback_exc).lower()
            if _is_ollama_unavailable_error(fallback_exc):
                raise LLMConfigurationError(_build_ollama_unavailable_message(base_url)) from fallback_exc
            if _is_ollama_memory_error(fallback_exc) or "not found" in message:
                continue
            raise

        os.environ["OLLAMA_MODEL"] = candidate_model
        os.environ["OLLAMA_ACTIVE_MODEL"] = candidate_model
        return response

    raise OllamaModelMemoryError(
        _build_memory_error_message(primary_model, attempted_models)
    )
