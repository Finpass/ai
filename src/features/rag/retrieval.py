import re
from dataclasses import dataclass

import psycopg

from src.core.config import settings
from src.features.rag import embedder, intent

_WORD = re.compile(r"[a-z0-9]{3,}")
RRF_K = 60


@dataclass
class Passage:
    chunk_id: int
    title: str
    source_type: str
    page: int
    section: str | None
    content: str
    key: str
    ticker: str | None
    score: float = 0.0


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal rank fusion: ids ranked high in several lists first."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda c: (-scores[c], c))


def or_query(text: str) -> str:
    words = list(dict.fromkeys(_WORD.findall(text.lower())))[:24]
    return " | ".join(words)


def known_tickers(conn: psycopg.Connection) -> set[str]:
    return {
        r[0]
        for r in conn.execute("SELECT DISTINCT ticker FROM rag_documents WHERE ticker IS NOT NULL")
    }


_FILTER = """d.status = 'active'
  AND (%(types)s::text[] IS NULL OR d.source_type = ANY(%(types)s))
  AND (%(tickers)s::text[] IS NULL OR d.ticker = ANY(%(tickers)s))"""


def candidates(
    conn: psycopg.Connection, query: str, types: list[str] | None, tickers: list[str] | None, k: int
) -> list[int]:
    vec = embedder.embed([query])[0]
    params = {"types": types, "tickers": tickers, "k": k}
    with conn.transaction():
        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        conn.execute("SET LOCAL hnsw.ef_search = 100")
        dense = [
            r[0]
            for r in conn.execute(
                f"SELECT c.id FROM rag_chunks c JOIN rag_documents d ON d.id = c.document_id WHERE {_FILTER} "
                "ORDER BY c.embedding <=> %(vec)s LIMIT %(k)s",
                {**params, "vec": vec},
            ).fetchall()
        ]
    sparse: list[int] = []
    if q := or_query(query):
        sparse = [
            r[0]
            for r in conn.execute(
                f"SELECT c.id FROM rag_chunks c JOIN rag_documents d ON d.id = c.document_id, to_tsquery('simple', %(q)s) tq "
                f"WHERE {_FILTER} AND c.tsv @@ tq ORDER BY ts_rank_cd(c.tsv, tq) DESC LIMIT %(k)s",
                {**params, "q": q},
            ).fetchall()
        ]
    return rrf([dense, sparse])[: settings.rag_top_k]


def load(conn: psycopg.Connection, ids: list[int]) -> list[Passage]:
    if not ids:
        return []
    rows = conn.execute(
        "SELECT c.id, d.title, d.source_type, c.page_start, c.section, c.content, d.r2_key, d.ticker "
        "FROM rag_chunks c JOIN rag_documents d ON d.id = c.document_id WHERE c.id = ANY(%s)",
        (ids,),
    ).fetchall()
    by = {r[0]: Passage(*r) for r in rows}
    return [by[i] for i in ids if i in by]


def dedupe_language(passages: list[Passage], lang: str) -> list[Passage]:
    """Same company-year can exist in both languages: keep the language of the question when both are present."""
    want = "idx_laporan_keuangan" if lang == "id" else "idx_financial_statement"
    other = "idx_financial_statement" if lang == "id" else "idx_laporan_keuangan"
    has_want = {(p.ticker, p.title.split(" ", 1)[0]) for p in passages if p.source_type == want}
    return [
        p
        for p in passages
        if not (p.source_type == other and (p.ticker, p.title.split(" ", 1)[0]) in has_want)
    ]


def search(
    conn: psycopg.Connection, message: str, history_query: str, lang: str, held_symbols: list[str]
) -> tuple[list[Passage], str, list[str]]:
    known = known_tickers(conn)
    mentioned = intent.mentioned_tickers(message, known)
    kind = intent.detect_intent(message, bool(mentioned))
    types = intent.source_types(kind, bool(mentioned))
    held = intent.held_tickers(held_symbols, known)
    tickers = mentioned or (held if kind == "perusahaan" else None) or None
    query = history_query or message
    ids = candidates(conn, query, types, tickers, settings.rag_top_k)
    passages = dedupe_language(load(conn, ids), lang)
    scores = embedder.rerank(query, [p.content for p in passages])
    for p, s in zip(passages, scores, strict=True):
        p.score = s + (0.05 if p.ticker in held else 0.0)
    passages.sort(key=lambda p: -p.score)
    keep = [p for p in passages[: settings.rag_top_n] if p.score >= settings.rag_min_score]
    return keep, kind, mentioned or held
