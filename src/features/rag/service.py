import asyncio
import logging
import uuid

from src.core.config import settings
from src.features.rag import guardrail, prompt, retrieval, store
from src.features.rag.retrieval import Passage
from src.features.rag.schemas import ChatRequest, ChatResponse, Citation, Trace, Turn_
from src.infrastructure.observability.turns import dump, turn_buffer
from src.infrastructure.openrouter import LLMError, OpenRouterClient

logger = logging.getLogger("rag.service")

NOT_FOUND = {
    "id": "Saya tidak menemukan informasi yang relevan di sumber yang tersedia (regulasi dan edukasi OJK, laporan keuangan emiten). Coba tanyakan dengan kata kunci yang lebih spesifik.",
    "en": "I could not find relevant information in the available sources (OJK regulation and education, issuer financial statements). Try a more specific question.",
}
FALLBACK_HEAD = {
    "id": "Saya belum bisa menyusun jawaban yang lolos pemeriksaan. Berikut kutipan paling relevan dari sumber:",
    "en": "I could not compose an answer that passed the checks. Here are the most relevant excerpts from the sources:",
}


def _query(req: ChatRequest) -> str:
    """Short follow-ups lean on the previous user turn so retrieval still has a topic."""
    if len(req.message.split()) < 5:
        prior = [t.content for t in req.history if t.role == "user"]
        if prior:
            return f"{prior[-1]} {req.message}"
    return req.message


def _search(req: ChatRequest) -> tuple[list[Passage], str, list[str]]:
    conn = store.connect()
    try:
        return retrieval.search(
            conn, req.message, _query(req), req.lang, [h.symbol for h in req.userContext.holdings]
        )
    finally:
        conn.close()


def _citations(passages: list[Passage], used: list[int]) -> list[Citation]:
    return [
        Citation(
            n=n,
            title=passages[n - 1].title,
            page=passages[n - 1].page,
            sourceType=passages[n - 1].source_type,
            key=passages[n - 1].key,
        )
        for n in used
    ]


def _snippet(content: str, limit: int = 240) -> str:
    """First readable stretch of a chunk (title lines dropped), cut at a word boundary."""
    body = " ".join(content.split("\n", 2)[-1].split())
    if len(body) <= limit:
        return body
    return body[:limit].rsplit(" ", 1)[0].rstrip(",;:-") + "..."


def _excerpts(passages: list[Passage], lang: str) -> str:
    lines = [FALLBACK_HEAD[lang], ""]
    for i, p in enumerate(passages[:3], 1):
        lines.append(f"- {p.title}, hal. {p.page}: {_snippet(p.content)} [{i}]")
    return "\n".join(lines)


TITLES = {
    "id": {
        "read": "Memahami pertanyaan...",
        "search": "Mencari di sumber OJK dan laporan keuangan...",
        "found": "{n} kutipan relevan dipilih",
        "none": "Tidak ada kutipan yang relevan",
        "write": "Menyusun jawaban...",
        "recheck": "Memeriksa ulang angka dan sitasi...",
        "done": "Jawaban siap",
        "fallback": "Jawaban disusun dari kutipan sumber",
        "error": "Gagal menyusun jawaban",
    },
    "en": {
        "read": "Understanding your question...",
        "search": "Searching OJK and financial statement sources...",
        "found": "{n} relevant excerpts selected",
        "none": "No relevant excerpts found",
        "write": "Writing the answer...",
        "recheck": "Re-checking numbers and citations...",
        "done": "Answer ready",
        "fallback": "Answer built from source excerpts",
        "error": "Could not compose an answer",
    },
}


async def chat(req: ChatRequest, llm: OpenRouterClient | None) -> ChatResponse:
    """Runs the pipeline and attaches the turn trail (also readable live through the turn buffer)."""
    run = req.runId or str(uuid.uuid4())
    title = TITLES[req.lang]
    try:
        resp = await _chat(req, llm, run, title)
    except Exception:
        turn_buffer.emit(run, title["error"], type="error", status="failed")
        raise
    resp.turns = [Turn_(**t) for t in dump(turn_buffer.read(run))]
    return resp


def _finish(run: str, title: dict[str, str], resp: ChatResponse) -> ChatResponse:
    if resp.source == "llm":
        turn_buffer.emit(run, title["done"], type="result", status="done")
    else:
        turn_buffer.emit(
            run, title["fallback"] if resp.grounded else title["none"], type="result", status="done"
        )
    return resp


async def _chat(
    req: ChatRequest, llm: OpenRouterClient | None, run: str, title: dict[str, str]
) -> ChatResponse:
    turn_buffer.emit(run, title["read"])
    turn_buffer.emit(run, title["search"])
    passages, intent_name, tickers = await asyncio.to_thread(_search, req)
    turn_buffer.emit(run, title["found"].format(n=len(passages)) if passages else title["none"])
    trace = Trace(
        intent=intent_name,
        tickers=tickers,
        chunkIds=[p.chunk_id for p in passages],
        topScore=round(passages[0].score, 3) if passages else 0.0,
    )
    has_facts = bool(req.userContext.profile or req.userContext.holdings)
    # nothing retrieved and the question is not about the user's own data: say so instead of guessing
    if not passages and not (intent_name == "portofolio" and has_facts):
        return _finish(
            run,
            title,
            ChatResponse(
                answer=NOT_FOUND[req.lang],
                citations=[],
                grounded=False,
                source="fallback",
                trace=trace,
            ),
        )
    if llm is None:
        return _finish(
            run,
            title,
            ChatResponse(
                answer=_excerpts(passages, req.lang),
                citations=_citations(passages, list(range(1, min(len(passages), 3) + 1))),
                grounded=bool(passages),
                source="fallback",
                trace=trace,
            ),
        )

    allowed = (
        prompt.facts_text(req.userContext)
        + "\n"
        + prompt.passages_text(passages)
        + "\n"
        + req.message
    )
    fix = ""
    for attempt in range(2):
        turn_buffer.emit(run, title["write"] if attempt == 0 else title["recheck"])
        try:
            answer = (
                await llm.complete(
                    prompt.build(req, passages, fix),
                    json_mode=False,
                    max_tokens=700,
                    timeout_seconds=settings.rag_chat_timeout_seconds,
                    model=settings.rag_model or None,
                )
            ).strip()
        except LLMError:
            break
        gap = guardrail.admits_gap(answer) and not guardrail.used_markers(answer, len(passages))
        bad = guardrail.violations(
            answer, allowed, len(passages), needs_citation=bool(passages) and not gap
        )
        if not bad:
            used = guardrail.used_markers(answer, len(passages))
            if gap:
                return _finish(
                    run,
                    title,
                    ChatResponse(
                        answer=answer, citations=[], grounded=False, source="llm", trace=trace
                    ),
                )
            return _finish(
                run,
                title,
                ChatResponse(
                    answer=answer,
                    citations=_citations(passages, used),
                    grounded=bool(passages),
                    source="llm",
                    trace=trace,
                ),
            )
        logger.info("guardrail rejected answer: %s", "; ".join(bad))
        fix = "; ".join(bad)
    if not passages:
        return _finish(
            run,
            title,
            ChatResponse(
                answer=NOT_FOUND[req.lang],
                citations=[],
                grounded=False,
                source="fallback",
                trace=trace,
            ),
        )
    return _finish(
        run,
        title,
        ChatResponse(
            answer=_excerpts(passages, req.lang),
            citations=_citations(passages, list(range(1, min(len(passages), 3) + 1))),
            grounded=True,
            source="fallback",
            trace=trace,
        ),
    )
