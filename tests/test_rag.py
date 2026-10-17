import asyncio

import pytest

from src.features.rag import guardrail, intent, meta, retrieval, service
from src.features.rag.chunking import chunk_pages
from src.features.rag.extract import Page
from src.features.rag.retrieval import Passage
from src.features.rag.schemas import ChatRequest, HoldingCtx, ProfileCtx, UserContext
from src.infrastructure.openrouter import LLMError


def test_parse_key_variants():
    assert meta.parse_key(
        "idx-dokumen/2024/Financial_Statement/AADI_FinancialStatement-2024-Tahunan-AADI.pdf"
    ) == meta.DocMeta(
        "idx_financial_statement", "AADI Financial Statement 2024", "AADI", 2024, "en"
    )
    m = meta.parse_key("idx-dokumen/2026/Laporan_Keuangan/BBCA_LaporanKeuangan-2026-II-BBCA.pdf")
    assert m and m.title == "BBCA Laporan Keuangan 2026 II" and m.language == "id"
    assert (
        meta.parse_key("idx-dokumen/2024/Laporan_Keuangan/AADI_FS 31 December 2024 AADI.pdf").ticker
        == "AADI"
    )
    assert meta.parse_key("ojk-regulasi/FAQ_x.pdf").source_type == "ojk_regulasi"
    assert (
        meta.parse_key("ojk-edukasi/101_101_2-9 Evaluation - K.pdf").title == "2-9 Evaluation - K"
    )
    assert meta.parse_key("misc/readme.pdf") is None


def test_chunking_keeps_pages_and_skips_empty():
    text = "Pasal 1\n" + " ".join(f"kata{i}" for i in range(900))
    pages = [
        Page(1, text, False),
        Page(2, "", False),
        Page(3, "isi halaman tiga yang cukup panjang untuk lolos batas minimal karakter", True),
    ]
    chunks = chunk_pages(pages, "ojk_regulasi", "Judul Dokumen")
    assert chunks and all(c.content.startswith("Judul Dokumen") for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert chunks[0].page_start == 1
    assert any(c.has_table for c in chunks)
    assert chunk_pages([Page(1, "", False)], "ojk_edukasi", "T") == []


def test_rrf_prefers_ids_present_in_both_lists():
    assert retrieval.rrf([[1, 2, 3], [3, 1, 4]])[:2] == [1, 3]
    assert retrieval.rrf([[], []]) == []


def test_or_query_sanitises():
    assert (
        retrieval.or_query("Apa itu 'POJK' & risiko; DROP--") == "apa | itu | pojk | risiko | drop"
    )


@pytest.mark.parametrize(
    ("msg", "tickers", "want"),
    [
        ("Bagaimana laba BBCA tahun lalu?", ["BBCA"], "perusahaan"),
        ("Apa aturan OJK soal pinjol?", [], "regulasi"),
        ("Jelaskan portofolio rekomendasi saya", [], "portofolio"),
        ("Apa itu diversifikasi?", [], "edukasi"),
    ],
)
def test_intent(msg, tickers, want):
    assert intent.detect_intent(msg, bool(tickers)) == want


def test_ticker_detection_needs_known_uppercase_symbol():
    known = {"BBCA", "TLKM"}
    assert intent.mentioned_tickers("laba BBCA.JK dan TLKM, bukan ABCD atau bbca", known) == [
        "BBCA",
        "TLKM",
    ]
    assert intent.held_tickers(["BBCA.JK", "AAPL", "tlkm.jk"], known) == ["BBCA", "TLKM"]


def test_guardrail_rules():
    ctx = "Laba bersih Rp 1.250 miliar [1] hal. 12; risiko moderat 3"
    ok = "Laba bersih Rp 1.250 miliar [1]."
    assert guardrail.violations(ok, ctx, 1, True) == []
    assert any(
        "not in context" in v
        for v in guardrail.violations("Laba Rp 9.999 miliar [1].", ctx, 1, True)
    )
    assert any(
        "advice" in v for v in guardrail.violations("Sebaiknya beli saham ini [1].", ctx, 1, True)
    )
    assert any("out of range" in v for v in guardrail.violations("Lihat [3].", ctx, 1, True))
    assert any(
        "missing citation" in v for v in guardrail.violations("Tanpa penanda.", ctx, 2, True)
    )
    assert guardrail.violations("Anda punya 3 aset.", ctx, 0, False) == []
    assert guardrail.used_markers("a [2] b [1] c [9] [2]", 3) == [1, 2]


def _passage(i=1, score=0.9):
    return Passage(
        i,
        "Abstrak PADK 1",
        "ojk_regulasi",
        2,
        None,
        "Judul\nBank wajib menerapkan manajemen risiko teknologi informasi 2026.",
        "ojk-regulasi/x.pdf",
        None,
        score,
    )


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    async def complete(self, messages, **kw):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _req(msg="Apa kewajiban bank soal teknologi informasi?", ctx=None):
    return ChatRequest(message=msg, userContext=ctx or UserContext())


def _patch(monkeypatch, passages, kind="regulasi"):
    monkeypatch.setattr(service, "_search", lambda req: (passages, kind, []))


def test_no_passages_is_not_grounded(monkeypatch):
    _patch(monkeypatch, [])
    out = asyncio.run(service.chat(_req(), FakeLLM([])))
    assert not out.grounded and out.source == "fallback" and not out.citations


def test_good_answer_cites_used_passages_only(monkeypatch):
    _patch(monkeypatch, [_passage(1), _passage(2)])
    llm = FakeLLM(["Bank wajib menerapkan manajemen risiko teknologi informasi [1]."])
    out = asyncio.run(service.chat(_req(), llm))
    assert (
        out.source == "llm"
        and out.grounded
        and [c.n for c in out.citations] == [1]
        and llm.calls == 1
    )


def test_retry_then_fallback_on_bad_answers(monkeypatch):
    _patch(monkeypatch, [_passage()])
    llm = FakeLLM(["Sebaiknya beli sekarang [1].", "Pasti untung 500 persen [1]."])
    out = asyncio.run(service.chat(_req(), llm))
    assert (
        out.source == "fallback"
        and llm.calls == 2
        and out.citations
        and "Abstrak PADK 1" in out.answer
    )


def test_retry_recovers(monkeypatch):
    _patch(monkeypatch, [_passage()])
    llm = FakeLLM(["Tanpa penanda.", "Bank wajib menerapkan manajemen risiko [1]."])
    assert asyncio.run(service.chat(_req(), llm)).source == "llm"


def test_llm_error_falls_back_to_excerpts(monkeypatch):
    _patch(monkeypatch, [_passage()])
    out = asyncio.run(service.chat(_req(), FakeLLM([LLMError("boom")])))
    assert out.source == "fallback" and out.citations


def test_portfolio_question_answers_from_facts_without_passages(monkeypatch):
    _patch(monkeypatch, [], kind="portofolio")
    ctx = UserContext(
        profile=ProfileCtx(riskProfile="Moderate", ladderLevel=3),
        holdings=[HoldingCtx(symbol="BBCA.JK", weightPct=40)],
    )
    llm = FakeLLM(["Profil Anda Moderate di level 3, dan BBCA.JK memegang 40% dari holdings."])
    out = asyncio.run(service.chat(_req("Jelaskan portofolio saya", ctx), llm))
    assert out.source == "llm" and not out.citations and llm.calls == 1


def test_extract_strips_nul_bytes(monkeypatch):
    from src.features.rag import extract

    class P:
        def extract_text(self):
            return "teks\x00 dengan\x00 NUL " * 5

    class Pdf:
        pages = [P()]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(extract.pdfplumber, "open", lambda _: Pdf())
    pages, _ = extract.extract_pages("x.pdf")
    assert "\x00" not in pages[0].text


def test_chat_attaches_turn_trail_and_buffer_serves_it(monkeypatch):
    from src.infrastructure.observability.turns import turn_buffer

    _patch(monkeypatch, [_passage()])
    req = ChatRequest(message="Apa kewajiban bank soal teknologi informasi?", runId="run-1")
    out = asyncio.run(service.chat(req, FakeLLM(["Bank wajib menerapkan manajemen risiko [1]."])))
    titles = [t.title for t in out.turns]
    assert titles[0].startswith("Memahami") and titles[-1] == "Jawaban siap"
    assert [t.sequence for t in out.turns] == list(range(1, len(titles) + 1))
    assert out.turns[-1].type == "result" and out.turns[-1].status == "done"
    assert [e.title for e in turn_buffer.read("run-1")] == titles
    assert [e.sequence for e in turn_buffer.read("run-1", since=2)][0] == 3


def test_chat_error_emits_failed_turn(monkeypatch):
    from src.infrastructure.observability.turns import turn_buffer

    def boom(_):
        raise RuntimeError("db down")

    monkeypatch.setattr(service, "_search", boom)
    with pytest.raises(RuntimeError):
        asyncio.run(service.chat(ChatRequest(message="halo", runId="run-err", lang="en"), None))
    last = turn_buffer.read("run-err")[-1]
    assert last.type == "error" and last.status == "failed"


def test_run_id_rejects_unsafe_characters():
    with pytest.raises(ValueError):
        ChatRequest(message="x", runId="../etc")


def test_gap_admission_is_a_valid_uncited_answer(monkeypatch):
    _patch(monkeypatch, [_passage()])
    llm = FakeLLM(["Informasi tentang hal itu tidak ditemukan di sumber yang tersedia."])
    out = asyncio.run(service.chat(_req(), llm))
    assert out.source == "llm" and not out.grounded and not out.citations and llm.calls == 1
    assert guardrail.admits_gap("I could not find this in the sources.")
    assert not guardrail.admits_gap("Bank wajib menerapkan manajemen risiko [1].")
