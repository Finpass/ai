import datetime
import os
from typing import Any

import chromadb

from src.core.config import settings
from src.domain.advisor_chat import intent
from src.domain.advisor_chat.documents import Passage
from src.infrastructure.advisor_chat import embedder

_chroma_client = None
_known_tickers_cache: set[str] | None = None


def get_collection():
    global _chroma_client
    if _chroma_client is None:
        chroma_path = os.getenv("CHROMA_PATH", "chroma_db")
        _chroma_client = chromadb.PersistentClient(path=chroma_path)
    return _chroma_client.get_collection("corpus_main")


def today_int() -> int:
    today = datetime.date.today()
    return today.year * 10000 + today.month * 100 + today.day


def map_doc_type(doc_type: str) -> str:
    if doc_type == "regulasi":
        return "ojk_regulasi"
    if doc_type == "edukasi":
        return "ojk_edukasi"
    if doc_type == "laporan_keuangan":
        return "idx_laporan_keuangan"
    return doc_type


def dedupe_language(passages: list[Passage], lang: str) -> list[Passage]:
    want = "idx_laporan_keuangan" if lang == "id" else "idx_financial_statement"
    other = "idx_financial_statement" if lang == "id" else "idx_laporan_keuangan"
    has_want = {(p.ticker, p.title.split(" ", 1)[0]) for p in passages if p.source_type == want}
    return [
        p
        for p in passages
        if not (p.source_type == other and (p.ticker, p.title.split(" ", 1)[0]) in has_want)
    ]


def known_tickers(conn: Any) -> set[str]:
    global _known_tickers_cache
    if _known_tickers_cache is not None:
        return _known_tickers_cache

    col = get_collection()
    unique_tickers = set()
    total = col.count()
    batch_size = 5000
    for offset in range(0, total, batch_size):
        data = col.get(include=["metadatas"], limit=batch_size, offset=offset)
        for m in data.get("metadatas", []):
            if m and m.get("ticker"):
                unique_tickers.add(m["ticker"])

    _known_tickers_cache = unique_tickers
    return _known_tickers_cache


def candidates(
    query: str, types: list[str] | None, tickers: list[str] | None, k: int
) -> list[Passage]:
    # TODO: Sisipkan penggabungan BM25 + RRF di sini nanti
    vec = embedder.embed([query])[0]

    filters: list[dict[str, Any]] = [
        {"status": "approved"},
        {"effective_date": {"$lte": today_int()}},
        {"expiry_date": {"$gte": today_int()}},
    ]

    if tickers:
        if len(tickers) == 1:
            filters.append({"ticker": tickers[0]})
        else:
            filters.append({"ticker": {"$in": tickers}})

    if types:
        mapped_types = []
        if "ojk_regulasi" in types:
            mapped_types.append("regulasi")
        if "ojk_edukasi" in types:
            mapped_types.append("edukasi")
        if "idx_laporan_keuangan" in types or "idx_financial_statement" in types:
            mapped_types.append("laporan_keuangan")

        if mapped_types:
            if len(mapped_types) == 1:
                filters.append({"doc_type": mapped_types[0]})
            else:
                filters.append({"doc_type": {"$in": mapped_types}})

    where_clause = {"$and": filters} if len(filters) > 1 else filters[0]

    col = get_collection()
    res = col.query(
        query_embeddings=[vec], n_results=k, where=where_clause, include=["documents", "metadatas"]
    )

    passages = []
    if not res.get("documents") or not res["documents"][0]:
        return passages

    for doc, m in zip(res["documents"][0], res["metadatas"][0]):
        if not m:
            continue
        passages.append(
            Passage(
                id=m.get("doc_id", ""),
                title=m.get("title", ""),
                source_type=map_doc_type(m.get("doc_type", "")),
                page_start=int(m.get("page", 0)),
                section=str(m.get("section", "")),
                content=doc,
                r2_key=str(m.get("r2_key", "")),
                ticker=m.get("ticker") if isinstance(m.get("ticker"), str) else None,
                tier=int(m.get("tier", 1)),
                nomor=str(m.get("nomor", "")),
                publisher=str(m.get("publisher", "OJK")),
                tanggal=int(m.get("published_date", 0) if m.get("published_date") else 0),
            )
        )
    return passages


def search(
    conn: Any, message: str, history_query: str, lang: str, held_symbols: list[str]
) -> tuple[list[Passage], str, list[str]]:
    known = known_tickers(conn)
    mentioned = intent.mentioned_tickers(message, known)
    kind = intent.detect_intent(message, bool(mentioned))
    types = intent.source_types(kind, bool(mentioned))
    held = intent.held_tickers(held_symbols, known)
    tickers = mentioned or (held if kind == "perusahaan" else None) or None

    query = history_query or message
    k = 50
    passages = dedupe_language(candidates(query, types, tickers, k), lang)

    if not passages:
        return [], kind, mentioned or held

    scores = embedder.rerank(query, [p.content for p in passages])
    for p, s in zip(passages, scores, strict=True):
        p.score = s + (0.05 if p.ticker and p.ticker in held else 0.0)

    passages.sort(key=lambda p: -p.score)
    keep = [p for p in passages[: settings.rag_top_n] if p.score >= settings.rag_min_score]

    return keep, kind, mentioned or held
