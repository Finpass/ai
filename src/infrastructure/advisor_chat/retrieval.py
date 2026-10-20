import datetime
import re
from typing import Any

import chromadb

from src.core.config import settings
from src.domain.advisor_chat import intent
from src.domain.advisor_chat.documents import Passage
from src.infrastructure.advisor_chat import embedder
from src.infrastructure.observability import tracing

_chroma_client = None
_known_tickers_cache: set[str] | None = None


def get_collection():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
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


def known_tickers() -> set[str]:
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


_YEAR = re.compile(r"\b(20[12]\d)\b")
# plain words in a question -> the statement names the tables actually use, so the dense search lands on them
_EXPAND = (
    (
        re.compile(
            r"\b(laba|rugi|profit|net income|earnings|pendapatan|revenue|penjualan|beban)\b", re.I
        ),
        "laporan laba rugi dan penghasilan komprehensif lain, penjualan, laba bruto, laba tahun berjalan, statement of profit or loss",
    ),
    (
        re.compile(r"\b(aset|aktiva|liabilitas|utang|ekuitas|asset|liabilit|equity)\w*", re.I),
        "laporan posisi keuangan, total aset, total liabilitas, total ekuitas, statement of financial position",
    ),
    (
        re.compile(r"\b(arus kas|cash flow|kas)\b", re.I),
        "laporan arus kas, arus kas dari aktivitas operasi, investasi, pendanaan, statement of cash flows",
    ),
)


def years_in(text: str) -> list[int]:
    return sorted({int(y) for y in _YEAR.findall(text)})


def expand(query: str) -> str:
    extra = [e for pat, e in _EXPAND if pat.search(query)]
    return " ".join([query, *extra])


def _candidates(
    query: str,
    types: list[str] | None,
    tickers: list[str] | None,
    k: int,
    year: int | None = None,
) -> list[Passage]:
    vec = embedder.embed([query], fast=True)[0]

    filters: list[dict[str, Any]] = [
        {"status": "approved"},
        {"expiry_date": {"$gte": today_int()}},
    ]

    if tickers:
        if len(tickers) == 1:
            filters.append({"ticker": tickers[0]})
        else:
            filters.append({"ticker": {"$in": tickers}})

    if year:
        filters.append({"tahun": year})

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
    tracing.current().metadata(where=where_clause)

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
                tahun=int(m.get("tahun") or 0),
            )
        )
    return passages


def _doc(p: Passage) -> dict[str, Any]:
    """A passage as the tracing helpers want it (source documents are public, so content is kept)."""
    return {
        "id": p.id, "score": round(p.score, 4) if p.score else None, "content": p.content,
        "title": p.title, "source_type": p.source_type, "page": p.page_start or None,
        "section": p.section or None, "ticker": p.ticker, "year": p.tahun or None,
    }


def candidates(
    query: str,
    types: list[str] | None,
    tickers: list[str] | None,
    k: int,
    year: int | None = None,
) -> list[Passage]:
    """Dense search in Chroma with metadata filters: one RETRIEVER span, the embedding inside it."""
    with tracing.span("vector_search", tracing.RETRIEVER, input=query) as t:
        t.metadata(k=k, year=year, tickers=tickers, source_types=types, collection="corpus_main")
        passages = _candidates(query, types, tickers, k, year)
        tracing.documents(t, "retrieval.documents", [_doc(p) for p in passages])
        t.metadata(returned=len(passages))
        return passages


def _gather(query: str, types, tickers, years: list[int], k: int) -> list[Passage]:
    """One search per year asked about, so a later year is not crowded out by the others; merged without repeats."""
    runs = (
        [candidates(query, types, tickers, k // len(years), y) for y in years]
        if years
        else [candidates(query, types, tickers, k)]
    )
    seen: set[str] = set()
    out: list[Passage] = []
    for p in (p for run in runs for p in run):
        if p.content not in seen:
            seen.add(p.content)
            out.append(p)
    return out


def search(
    message: str, history_query: str, lang: str, held_symbols: list[str]
) -> tuple[list[Passage], str, list[str]]:
    with tracing.span("plan_retrieval", tracing.CHAIN, input=history_query or message) as plan:
        known = known_tickers()
        mentioned = intent.mentioned_tickers(message, known)
        kind = intent.detect_intent(message, bool(mentioned))
        types = intent.source_types(kind, bool(mentioned))
        held = intent.held_tickers(held_symbols, known)
        tickers = mentioned or (held if kind == "perusahaan" else None) or None

        query = expand(history_query or message)
        # years only narrow company filings; regulation and education documents are not filed per year
        years = years_in(message) if kind == "perusahaan" else []
        plan.metadata(
            intent=kind, tickers=tickers, mentioned_tickers=mentioned, held_tickers=held,
            source_types=types, years=years, query_expanded=query != (history_query or message),
            followup_query=bool(history_query and history_query != message), lang=lang,
        )
        plan.output(
            {"intent": kind, "tickers": tickers, "source_types": types, "years": years, "query": query}
        )

    with tracing.span("gather_candidates", tracing.CHAIN, input=query) as gather:
        raw = _gather(query, types, tickers, years, settings.rag_top_k)
        passages = dedupe_language(raw, lang)
        gather.metadata(top_k=settings.rag_top_k, searches=max(len(years), 1), candidates=len(raw),
                        after_language_dedupe=len(passages))
        gather.output([{"id": p.id, "title": p.title, "score": round(p.score, 4)} for p in passages])

    if not passages:
        tracing.current().metadata(no_candidates=True)
        return [], kind, mentioned or held

    with tracing.span(
        "rerank_passages",
        tracing.RERANKER,
        attributes={
            "reranker.model_name": settings.rerank_model,
            "reranker.top_k": settings.rag_top_n,
            **({"reranker.query": query} if tracing.capture_content() else {}),
        },
    ) as rr:
        scores = embedder.rerank(query, [p.content for p in passages])
        tracing.documents(rr, "reranker.input_documents", [_doc(p) for p in passages])
        for p, s in zip(passages, scores, strict=True):
            p.score = s + (0.05 if p.ticker and p.ticker in held else 0.0)
        passages.sort(key=lambda p: -p.score)
        top = passages[: settings.rag_top_n * 2]
        tracing.documents(rr, "reranker.output_documents", [_doc(p) for p in top])
        rr.metadata(
            candidates=len(passages), held_boost=0.05, best_score=round(passages[0].score, 4)
        )

    with tracing.span("select_passages", tracing.CHAIN, input=query) as pick:
        passages = [p for p in passages if p.score >= settings.rag_min_score]
        above_cutoff = len(passages)

        keep: list[Passage] = []
        if years:  # best few of every year asked about first, so each year is represented
            per_year = max(2, settings.rag_top_n // len(years))
            for y in years:
                keep += [p for p in passages if p.tahun == y][:per_year]
        keep += [p for p in passages if p not in keep]
        keep = keep[: settings.rag_top_n]
        keep.sort(key=lambda p: -p.score)
        pick.metadata(min_score=settings.rag_min_score, above_cutoff=above_cutoff, kept=len(keep),
                      top_n=settings.rag_top_n, years_balanced=bool(years),
                      top_score=round(keep[0].score, 4) if keep else None)
        pick.output([{"id": p.id, "title": p.title, "score": round(p.score, 4)} for p in keep])

    return keep, kind, mentioned or held
