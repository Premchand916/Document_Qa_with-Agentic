"""Fingerprint the loaded embedding coordinate system before serving an index."""

import hashlib
import json


def _stable_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _model_weights_hash(client):
    import torch

    cached = getattr(client, "_docqa_weights_hash", None)
    if cached:
        return cached
    digest = hashlib.sha256()
    for name, tensor in sorted(client.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_stable_json(list(tensor.shape)))
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    value = digest.hexdigest()
    client._docqa_weights_hash = value
    return value


def embedding_identity(embedding) -> dict:
    """Return dimension and a digest of model config, tokenizer, and loaded weights.

    Generic embedding fixtures use public scalar configuration. Production
    HuggingFaceEmbeddings include the loaded SentenceTransformer state dict.
    """
    config = {"class": f"{type(embedding).__module__}.{type(embedding).__qualname__}"}
    client = getattr(embedding, "client", None)
    if client is not None and hasattr(client, "state_dict"):
        model_kwargs = dict(getattr(embedding, "model_kwargs", {}) or {})
        model_kwargs.pop("local_files_only", None)
        tokenizer = getattr(client, "tokenizer", None)
        tokenizer_hash = None
        if tokenizer is not None:
            tokenizer_hash = hashlib.sha256(_stable_json({
                "vocab": tokenizer.get_vocab(),
                "special_tokens": tokenizer.special_tokens_map,
            })).hexdigest()
        modules = getattr(client, "_modules", {})
        config.update(
            model_name=getattr(embedding, "model_name", None),
            model_kwargs=model_kwargs,
            encode_kwargs=getattr(embedding, "encode_kwargs", {}),
            query_encode_kwargs=getattr(embedding, "query_encode_kwargs", {}),
            tokenizer_name=getattr(tokenizer, "name_or_path", None),
            tokenizer_vocab_size=getattr(tokenizer, "vocab_size", None),
            tokenizer_sha256=tokenizer_hash,
            modules=[f"{name}:{type(module).__module__}.{type(module).__qualname__}" for name, module in modules.items()],
            weights_sha256=_model_weights_hash(client),
        )
        dimension = client.get_sentence_embedding_dimension()
    else:
        config["settings"] = {name: value for name, value in vars(embedding).items()
                              if not name.startswith("_") and isinstance(value, (str, int, float, bool, type(None)))}
        dimension = len(embedding.embed_query("docqa dimension probe"))
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("Embedding dimension is unavailable")
    return {"dimension": dimension, "fingerprint": hashlib.sha256(_stable_json(config)).hexdigest()}
