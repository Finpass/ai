import logging
import math
import os
import threading
import time

import httpx
import numpy as np

from src.core.config import settings
from src.infrastructure.observability import tracing

logger = logging.getLogger("rag.models")
_lock = threading.Lock()
_models: dict[str, object] = {}

EMBED_MODEL = os.environ.get("EMBED_MODEL", "").strip() or "baai/bge-m3"
EMBED_DIMENSIONS = 1024
OPENROUTER_BASE_URL = (
    os.environ.get("OPENROUTER_BASE_URL", "").strip() or "https://openrouter.ai/api/v1"
)
MAX_ATTEMPTS = 5
REQUEST_TIMEOUT_SECONDS = 30.0


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _reranker():
    with _lock:
        if "rerank" not in _models:
            from sentence_transformers import CrossEncoder

            logger.info("loading %s", settings.rerank_model)
            _models["rerank"] = CrossEncoder(
                settings.rerank_model, device=_device(), max_length=512
            )
        return _models["rerank"]


def _api_key() -> str:
    # pydantic-settings membaca .env tanpa mengisi os.environ, jadi cek keduanya
    value = getattr(settings, "openrouter_api_key", None) or os.environ.get(
        "OPENROUTER_API_KEY", ""
    )
    if hasattr(value, "get_secret_value"):
        value = value.get_secret_value()
    key = str(value).strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY belum diisi")
    return key


def _normalize(vector: list[float]) -> list[float]:
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm == 0.0:
        raise RuntimeError("vektor embedding nol atau tidak valid")
    return (array / norm).tolist()


def _parse(batch: list[str], body: object) -> list[list[float]]:
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or len(data) != len(batch):
        raise RuntimeError("jumlah vektor tidak sama dengan jumlah input")
    vectors = []
    for item in sorted(data, key=lambda entry: entry.get("index", 0)):
        raw = item.get("embedding")
        if not isinstance(raw, list) or len(raw) != EMBED_DIMENSIONS:
            raise RuntimeError(f"dimensi embedding tidak sesuai, diharapkan {EMBED_DIMENSIONS}")
        vectors.append(_normalize(raw))
    return vectors


def _embed_batch(
    http: httpx.Client, batch: list[str], attempts: int = MAX_ATTEMPTS
) -> list[list[float]]:
    url = OPENROUTER_BASE_URL.rstrip("/") + "/embeddings"
    headers = {"Authorization": f"Bearer {_api_key()}"}
    payload = {"model": EMBED_MODEL, "input": batch, "encoding_format": "float"}
    for attempt in range(attempts):
        try:
            response = http.post(url, headers=headers, json=payload)
        except httpx.TransportError:
            response = None

        if response is not None and response.status_code < 400:
            body = response.json()
            vectors = _parse(batch, body)
            tracing.current().usage(body.get("usage") if isinstance(body, dict) else None, kind=tracing.EMBEDDING)
            return vectors

        retryable = response is None or response.status_code == 429 or response.status_code >= 500
        if not retryable or attempt == attempts - 1:
            detail = "koneksi gagal" if response is None else f"HTTP {response.status_code}"
            raise RuntimeError(f"embedding gagal: {detail}")

        time.sleep(min(2.0**attempt, 30.0))
    raise RuntimeError("tidak tercapai")


def embed(texts: list[str], batch_size: int = 16, *, fast: bool = False) -> list[list[float]]:
    # Jika fast=True, batasi maksimal 2 attempt dan timeout 10 detik
    attempts, timeout = (2, 10.0) if fast else (MAX_ATTEMPTS, REQUEST_TIMEOUT_SECONDS)
    vectors: list[list[float]] = []

    with tracing.embedding_span(texts, model=EMBED_MODEL, dimensions=EMBED_DIMENSIONS) as t:
        t.metadata(fast=fast, batch_size=batch_size)
        with httpx.Client(timeout=timeout) as http:
            for start in range(0, len(texts), batch_size):
                vectors.extend(_embed_batch(http, texts[start : start + batch_size], attempts))
        t.ok()

    return vectors


def rerank(query: str, passages: list[str]) -> list[float]:
    if not passages:
        return []
    scores = _reranker().predict([(query, p) for p in passages], batch_size=8, activation_fct=None)
    return [float(1 / (1 + np.exp(-s))) for s in scores]


def warmup() -> None:
    embed(["warmup"])
    rerank("warmup", ["warmup"])
