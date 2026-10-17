from src.features.rag.retrieval import Passage
from src.features.rag.schemas import ChatRequest, UserContext

SYSTEM = {
    "id": (
        "Kamu adalah asisten edukasi keuangan Finpass. Jawab dalam bahasa Indonesia yang sederhana dan ringkas.\n"
        "Aturan:\n"
        "1. Jawab hanya dari FAKTA USER dan KUTIPAN SUMBER di bawah. Jangan memakai pengetahuan lain untuk angka atau klaim.\n"
        "2. Setiap klaim dari sumber diberi penanda [n] sesuai nomor kutipan. Jangan menulis nomor yang tidak ada.\n"
        "3. Salin angka persis seperti di fakta atau kutipan. Jangan menghitung ulang atau membulatkan.\n"
        "4. Jangan memberi saran beli, jual, atau tahan, dan jangan menjanjikan hasil. Jelaskan saja, termasuk risiko dan trade-off.\n"
        "5. Jika sumber tidak cukup, katakan terus terang bahwa informasinya tidak ditemukan di sumber.\n"
        "6. Jika pertanyaan tentang profil atau portofolio user, jelaskan dari FAKTA USER tanpa penanda sumber.\n"
        'Format: paragraf pendek; untuk beberapa poin pakai baris diawali "- " (satu poin per baris); letakkan penanda [n] di akhir kalimat. '
        "Tulis teks biasa, tanpa markdown tebal dan tanpa emoji."
    ),
    "en": (
        "You are Finpass's financial education assistant. Answer in plain, concise English.\n"
        "Rules:\n"
        "1. Answer only from USER FACTS and SOURCE EXCERPTS below. Do not use outside knowledge for numbers or claims.\n"
        "2. Mark every claim taken from a source with [n] matching the excerpt number. Never write a number that does not exist.\n"
        "3. Copy numbers exactly as in the facts or excerpts. Do not recompute or round.\n"
        "4. Never advise to buy, sell, or hold, and never promise results. Just explain, including risks and trade-offs.\n"
        "5. If the sources are not enough, say plainly that the information was not found in the sources.\n"
        "6. If the question is about the user's profile or portfolio, explain from USER FACTS without source markers.\n"
        'Format: short paragraphs; for several points use lines starting with "- " (one point per line); put [n] at the end of the sentence. '
        "Plain text only, no bold markdown, no emoji."
    ),
}


def facts_text(ctx: UserContext) -> str:
    lines: list[str] = []
    if p := ctx.profile:
        lines.append(
            f"Profil: risiko {p.riskProfile}, level tangga {p.ladderLevel}, syariah {p.sharia}, horizon {p.horizonYears:g} tahun"
            + (f", tujuan {p.goal}" if p.goal else "")
        )
    if ctx.holdings:
        lines.append(
            "Holdings: "
            + "; ".join(f"{h.symbol} ({h.assetClass}) {h.weightPct:g}%" for h in ctx.holdings[:25])
        )
    if ctx.recommended:
        parts = []
        for c in ctx.recommended:
            picks = ", ".join(f"{a.ticker} {a.weightPct:g}%" for a in c.assets if a.ticker)
            parts.append(f"{c.cls} {c.allocationPct:g}%" + (f" [{picks}]" if picks else ""))
        lines.append(
            "Portofolio rekomendasi (pencocokan profil, bukan ajakan membeli): " + "; ".join(parts)
        )
    return "\n".join(lines) or "(tidak ada)"


def passages_text(passages: list[Passage]) -> str:
    if not passages:
        return "(tidak ada kutipan relevan)"
    return "\n\n".join(
        f"[{i}] {p.title}, hal. {p.page}\n{p.content[:1800]}" for i, p in enumerate(passages, 1)
    )


def build(req: ChatRequest, passages: list[Passage], fix: str = "") -> list[dict[str, str]]:
    system = (
        SYSTEM[req.lang]
        + f"\n\nFAKTA USER:\n{facts_text(req.userContext)}\n\nKUTIPAN SUMBER:\n{passages_text(passages)}"
    )
    msgs = [{"role": "system", "content": system}]
    msgs += [{"role": t.role, "content": t.content} for t in req.history[-6:]]
    msgs.append(
        {
            "role": "user",
            "content": req.message + (f"\n\n(Perbaiki jawabanmu: {fix})" if fix else ""),
        }
    )
    return msgs
