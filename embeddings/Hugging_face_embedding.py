import os
from functools import lru_cache
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

MODEL_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"


def _env_flag(name):
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    return raw_value.lower() in {"1", "true", "yes"}


def _cached_model_exists(model_name):
    snapshot_dir = MODEL_CACHE_DIR / f"models--{model_name.replace('/', '--')}"
    return snapshot_dir.exists()


def _should_use_local_files_only(model_name):
    configured = _env_flag("HF_LOCAL_FILES_ONLY")
    if configured is not None:
        return configured

    # Default to offline-first behavior. Users can opt into downloads with
    # HF_LOCAL_FILES_ONLY=false when they actually want network fetches.
    return True


@lru_cache(maxsize=1)
def get_embedding_model():
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    local_files_only = _should_use_local_files_only(model_name)

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=str(MODEL_CACHE_DIR),
            model_kwargs={"local_files_only": local_files_only}
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to initialize embedding model '{model_name}'. "
            f"Expected cache folder: {MODEL_CACHE_DIR}. "
            "Download the model into that folder, or set EMBEDDING_MODEL to a local cached model path. "
            "If you want automatic downloads, set HF_LOCAL_FILES_ONLY=false before starting the app."
        ) from exc

    return embeddings
