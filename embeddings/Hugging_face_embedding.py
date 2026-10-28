import os
from functools import lru_cache
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

MODEL_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"
DEFAULT_EMBEDDING_BATCH_SIZE = 32
MAX_EMBEDDING_BATCH_SIZE = 256


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


def _embedding_batch_size() -> int:
    raw_value = os.getenv("EMBEDDING_BATCH_SIZE", str(DEFAULT_EMBEDDING_BATCH_SIZE)).strip()
    try:
        batch_size = int(raw_value)
    except ValueError as exc:
        raise ValueError("EMBEDDING_BATCH_SIZE must be an integer") from exc
    if not 1 <= batch_size <= MAX_EMBEDDING_BATCH_SIZE:
        raise ValueError(f"EMBEDDING_BATCH_SIZE must be between 1 and {MAX_EMBEDDING_BATCH_SIZE}")
    return batch_size


@lru_cache(maxsize=4)
def _load_embedding_model(model_name, local_files_only, revision, batch_size=DEFAULT_EMBEDDING_BATCH_SIZE):
    try:
        model_kwargs = {"local_files_only": local_files_only}
        if revision:
            model_kwargs["revision"] = revision
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=str(MODEL_CACHE_DIR) if _cached_model_exists(model_name) else None,
            model_kwargs=model_kwargs,
            encode_kwargs={"batch_size": batch_size},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to initialize embedding model '{model_name}'. "
            f"Checked project cache {MODEL_CACHE_DIR} and the standard Hugging Face cache. "
            "Cache the model locally, or set EMBEDDING_MODEL to a local model path. "
            "If you want automatic downloads, set HF_LOCAL_FILES_ONLY=false before starting the app."
        ) from exc

    return embeddings


def get_embedding_model():
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return _load_embedding_model(model_name, _should_use_local_files_only(model_name),
                                 os.getenv("EMBEDDING_REVISION", ""), _embedding_batch_size())


get_embedding_model.cache_clear = _load_embedding_model.cache_clear
