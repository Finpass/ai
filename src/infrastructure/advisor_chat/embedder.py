import logging
import threading

import numpy as np

from src.core.config import settings

logger = logging.getLogger("rag.models")
_lock = threading.Lock()
_models: dict[str, object] = {}


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _embedder():
    with _lock:
        if "embed" not in _models:
            from sentence_transformers import SentenceTransformer

            logger.info("loading %s", settings.embed_model)
            m = SentenceTransformer(settings.embed_model, device=_device())
            m.max_seq_length = 512
            _models["embed"] = m
        return _models["embed"]


def _reranker():
    with _lock:
        if "rerank" not in _models:
            from sentence_transformers import CrossEncoder

            logger.info("loading %s", settings.rerank_model)
            _models["rerank"] = CrossEncoder(
                settings.rerank_model, device=_device(), max_length=512
            )
        return _models["rerank"]


def embed(texts: list[str], batch_size: int = 16) -> np.ndarray:
    m = _embedder()
    return m.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )  # type: ignore[attr-defined]


def rerank(query: str, passages: list[str]) -> list[float]:
    if not passages:
        return []
    scores = _reranker().predict([(query, p) for p in passages], batch_size=8, activation_fct=None)  # type: ignore[attr-defined]
    return [float(1 / (1 + np.exp(-s))) for s in scores]  # sigmoid: 0..1


def warmup() -> None:
    embed(["warmup"])
    rerank("warmup", ["warmup"])
