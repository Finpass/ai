import asyncio
import logging
import re
import uuid

from src.core.config import settings
from src.domain.advisor_chat import facts, guardrail
from src.domain.advisor_chat.documents import Passage
from src.domain.advisor_chat.schemas import ChatRequest, ChatResponse, Citation, Trace, Turn_
from src.infrastructure.advisor_chat import retrieval
from src.infrastructure.observability.narrator import Narrator
from src.infrastructure.observability.turns import dump, turn_buffer
from src.infrastructure.openrouter import LLMError, OpenRouterClient
from src.infrastructure.prompt.loader import load

logger = logging.getLogger("rag.service")

NEWS_WHEN_DOCS_BELOW = 3

NOT_FOUND = {
    "id": "Saya tidak menemukan informasi yang relevan di sumber yang tersedia (regulasi dan edukasi OJK, laporan keuangan emiten). Coba tanyakan dengan kata kunci yang lebih spesifik.",
    "en": "I could not find relevant information in the available sources (OJK regulation and education, issuer financial statements). Try a more specific question.",
}
FALLBACK_HEAD = {
    "id": "Saya belum bisa menyusun jawaban yang lolos pemeriksaan. Berikut kutipan paling relevan dari sumber:",
    "en": "I could not compose an answer that passed the checks. Here are the most relevant excerpts from the sources:",
}


_TAGS = re.compile(r"</?\s*(?:fakta|sumber)\s*>", re.IGNORECASE)


def _fence(text: str) -> str:
    """Untrusted text (news, documents) cannot close or reopen the blocks the prompt tells the model to treat as data."""
    return _TAGS.sub("", text)


def _messages(req: ChatRequest, passages: list[Passage], fix: str = "") -> list[dict[str, str]]:
    system = (
        load(f"advisor_chat_system_{req.lang}")
        + f"\n\nFAKTA USER:\n<fakta>\n{_fence(facts.facts_text(req.userContext))}\n</fakta>"
        + f"\n\nKUTIPAN SUMBER:\n<sumber>\n{_fence(facts.passages_text(passages))}\n</sumber>"
    )
    msgs = [{"role": "system", "content": system}]
    msgs += [{"role": t.role, "content": _strip_note(t.content)} for t in req.history[-6:]]
    suffix = f"\n\n(Perbaiki jawabanmu: {fix})" if fix else ""
    msgs.append({"role": "user", "content": req.message + suffix})
    return msgs


def _query(req: ChatRequest) -> str:
    """Short follow-ups lean on the previous user turn so retrieval still has a topic."""
    if len(req.message.split()) < 5:
        prior = [t.content for t in req.history if t.role == "user"]
        if prior:
            return f"{prior[-1]} {req.message}"
    return req.message


def _search(req: ChatRequest) -> tuple[list[Passage], str, list[str]]:
    return retrieval.search(
        req.message, _query(req), req.lang, [h.symbol for h in req.userContext.holdings]
    )


def _citations(passages: list[Passage], used: list[int]) -> list[Citation]:
    return [
        Citation(
            n=n,
            title=passages[n - 1].title,
            page=passages[n - 1].page_start or None,
            sourceType=passages[n - 1].source_type,
            key=passages[n - 1].r2_key,
            url=passages[n - 1].r2_key if passages[n - 1].r2_key.startswith("http") else None,
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
        page = f", hal. {p.page_start}" if p.page_start else ""
        lines.append(f"- {p.title}{page}: {_snippet(p.content)} [{i}]")
    return _with_note("\n".join(lines), lang)


def _strip_note(text: str) -> str:
    """Drops the fixed disclaimer: stored turns carry it, and a model that sees it in history copies it."""
    for lang in ("id", "en"):
        text = text.replace(load(f"advisor_chat_disclaimer_{lang}"), "")
    return text.strip()


def _with_note(answer: str, lang: str) -> str:
    """Every answer ends with the same fixed disclaimer, written by us and not by the model."""
    return f"{_strip_note(answer)}\n\n{load(f'advisor_chat_disclaimer_{lang}')}"


# plain fallbacks: used when the model cannot write a title, or is not configured at all
TITLES = {
    "id": {
        "read": "Memahami pertanyaan...",
        "search": "Mencari di sumber OJK dan laporan keuangan...",
        "found": "{n} kutipan relevan dipilih",
        "none": "Tidak ada kutipan yang relevan",
        "news": "Mempertimbangkan {n} berita terbaru",
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
        "news": "Considering {n} recent news items",
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
    nar = Narrator(run, req.lang, req.message, llm)
    try:
        resp = await _chat(req, llm, nar)
    except Exception:
        await nar.drain()
        turn_buffer.emit(run, TITLES[req.lang]["error"], type="error", status="failed")
        raise
    await nar.drain()
    resp.turns = [Turn_(**t) for t in dump(turn_buffer.read(run))]
    return resp


def _finish(nar: Narrator, lang: str, resp: ChatResponse) -> ChatResponse:
    t = TITLES[lang]
    if resp.source == "llm":
        nar.say(
            "done", "jawaban selesai dan lolos pemeriksaan", t["done"], type="result", status="done"
        )
    else:
        kind = "fallback" if resp.grounded else "none"
        nar.say(
            "fallback",
            "jawaban disusun dari kutipan sumber" if resp.grounded else "tidak ada kutipan relevan",
            t[kind],
            type="result",
            status="done",
        )
    return resp


async def _chat(req: ChatRequest, llm: OpenRouterClient | None, nar: Narrator) -> ChatResponse:
    t = TITLES[req.lang]
    nar.say("read", "pertanyaan pengguna diterima", t["read"])
    nar.say(
        "search", "mencari di regulasi dan edukasi OJK serta laporan keuangan emiten", t["search"]
    )
    docs, intent_name, tickers = await asyncio.to_thread(_search, req)
    if docs:
        names = "; ".join(p.title for p in docs[:3])
        nar.say(
            "found",
            f"{len(docs)} kutipan dokumen dipilih, teratas: {names}",
            t["found"].format(n=len(docs)),
        )
    else:
        nar.say("found", "tidak ada dokumen yang cocok dengan pertanyaan", t["none"])
    news = facts.news_passages(req.userContext.news) if len(docs) < NEWS_WHEN_DOCS_BELOW else []
    if news:
        names = "; ".join(p.title for p in news[:3])
        nar.say(
            "news",
            f"dokumen kurang, memakai {len(news)} berita terbaru: {names}",
            t["news"].format(n=len(news)),
        )
    passages = docs + news
    trace = Trace(
        intent=intent_name,
        tickers=tickers,
        chunkIds=[str(p.id) for p in docs],
        topScore=round(docs[0].score, 3) if docs else 0.0,
    )
    has_facts = bool(
        req.userContext.profile or req.userContext.holdings or req.userContext.comparison
    )
    # nothing retrieved and the question is not about the user's own data: say so instead of guessing
    if not passages and not (
        (intent_name == "portofolio" or req.userContext.comparison) and has_facts
    ):
        return _finish(
            nar,
            req.lang,
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
            nar,
            req.lang,
            ChatResponse(
                answer=_excerpts(passages, req.lang),
                citations=_citations(passages, list(range(1, min(len(passages), 3) + 1))),
                grounded=bool(passages),
                source="fallback",
                trace=trace,
            ),
        )

    fix = ""
    for attempt in range(2):
        if attempt == 0:
            used = f"{len(docs)} dokumen, {len(news)} berita, konteks personal user"
            nar.say("write", f"menyusun jawaban dari {used}", t["write"])
        else:
            nar.say("recheck", f"jawaban pertama ditolak pemeriksaan: {fix}", t["recheck"])
        try:
            answer = (
                await llm.complete(
                    _messages(req, passages, fix),
                    json_mode=False,
                    max_tokens=1200,
                    timeout_seconds=settings.rag_chat_timeout_seconds,
                    model=settings.rag_model or None,
                )
            ).strip()
        except LLMError:
            break
        gap = guardrail.admits_gap(answer) and not guardrail.used_markers(answer, len(passages))
        bad = guardrail.violations(answer, len(passages))
        if not bad:
            used_n = guardrail.used_markers(answer, len(passages))
            answer = _with_note(answer, req.lang)
            if gap:
                return _finish(
                    nar,
                    req.lang,
                    ChatResponse(
                        answer=answer, citations=[], grounded=False, source="llm", trace=trace
                    ),
                )
            return _finish(
                nar,
                req.lang,
                ChatResponse(
                    answer=answer,
                    citations=_citations(passages, used_n),
                    grounded=bool(passages),
                    source="llm",
                    trace=trace,
                ),
            )
        logger.info("guardrail rejected answer: %s", "; ".join(bad))
        fix = "; ".join(bad)
    if not passages:
        return _finish(
            nar,
            req.lang,
            ChatResponse(
                answer=NOT_FOUND[req.lang],
                citations=[],
                grounded=False,
                source="fallback",
                trace=trace,
            ),
        )
    return _finish(
        nar,
        req.lang,
        ChatResponse(
            answer=_excerpts(passages, req.lang),
            citations=_citations(passages, list(range(1, min(len(passages), 3) + 1))),
            grounded=True,
            source="fallback",
            trace=trace,
        ),
    )
