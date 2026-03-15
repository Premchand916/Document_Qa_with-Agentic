import os
from functools import lru_cache
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

MODEL_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"
LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "true").lower() in {"1", "true", "yes"}


@lru_cache(maxsize=1)
def get_embedding_model():
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=str(MODEL_CACHE_DIR),
            model_kwargs={"local_files_only": LOCAL_FILES_ONLY}
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to initialize embedding model '{model_name}'. "
            f"Expected cache folder: {MODEL_CACHE_DIR}. "
            "Download the model into that folder, or set EMBEDDING_MODEL to a local cached model path. "
            "If you want automatic downloads, set HF_LOCAL_FILES_ONLY=false before starting the app."
        ) from exc

    return embeddings
