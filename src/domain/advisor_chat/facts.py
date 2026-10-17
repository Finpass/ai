from src.domain.advisor_chat.documents import Passage
from src.domain.advisor_chat.schemas import NewsCtx, ProfileCtx, UserContext


def _idr(v: int) -> str:
    return "Rp " + f"{v:,}".replace(",", ".")


def _profile_lines(p: ProfileCtx) -> list[str]:
    head = (
        f"Profil: risiko {p.riskProfile}, level tangga {p.ladderLevel}, syariah {p.sharia}, "
        f"horizon {p.horizonYears:g} tahun"
    )
    if p.goal:
        head += f", tujuan {p.goal}"
    if p.goalAmountIDR:
        head += f" senilai {_idr(p.goalAmountIDR)}"
    lines = [head]
    money = [
        ("pendapatan per bulan", p.incomeIDR),
        ("pengeluaran per bulan", p.expensesIDR),
        ("surplus per bulan", p.surplusIDR),
        ("cicilan per bulan", p.installmentIDR),
        ("tabungan", p.savingsIDR),
        ("dana yang bisa diinvestasikan", p.investableIDR),
    ]
    parts = [f"{k} {_idr(v)}" for k, v in money if v is not None]
    if p.dependents is not None:
        parts.append(f"tanggungan {p.dependents} orang")
    if p.employment:
        parts.append(f"pekerjaan {p.employment}")
    if p.knowledge:
        parts.append(f"pengetahuan investasi {p.knowledge}")
    if parts:
        lines.append("Keuangan: " + ", ".join(parts))
    if p.emergencyMonths is not None:
        target = (
            f" dari target {p.emergencyTargetMonths:g} bulan" if p.emergencyTargetMonths else ""
        )
        lines.append(f"Dana darurat: {p.emergencyMonths:g} bulan pengeluaran{target}")
    if p.healthCategory:
        score = f" (skor {p.healthScore:g})" if p.healthScore is not None else ""
        lines.append(f"Kesehatan finansial: {p.healthCategory}{score}")
    if p.heldKinds:
        lines.append("Jenis aset yang pernah dipegang: " + ", ".join(p.heldKinds))
    if p.priorities:
        lines.append("Prioritas: " + ", ".join(p.priorities))
    return lines


def facts_text(ctx: UserContext) -> str:
    lines: list[str] = []
    if ctx.profile:
        lines += _profile_lines(ctx.profile)
    if ctx.holdings:
        lines.append(
            "Holdings: "
            + "; ".join(
                f"{h.symbol or h.name} ({h.assetClass}) {h.weightPct:g}%"
                + (f", nilai {_idr(h.valueIDR)}" if h.valueIDR else "")
                for h in ctx.holdings[:25]
            )
        )
    if ctx.recommended:
        parts = []
        for c in ctx.recommended:
            picks = ", ".join(f"{a.ticker} {a.weightPct:g}%" for a in c.assets if a.ticker)
            parts.append(f"{c.cls} {c.allocationPct:g}%" + (f" [{picks}]" if picks else ""))
        lines.append(
            "Portofolio rekomendasi (pencocokan profil, bukan ajakan membeli): " + "; ".join(parts)
        )
    if ctx.comparison:
        lines.append("Perbandingan produk (hasil aturan profil, bukan ajakan membeli):")
        for cmp in ctx.comparison:
            status = (
                f"lolos gate, peringkat {cmp.rank}" if cmp.eligible else f"tidak lolos {cmp.gate}"
            )
            extra = "".join(
                f", {k} {v:g}"
                for k, v in (
                    ("sharpe", cmp.sharpe),
                    ("dividen", cmp.divYield),
                    ("PER", cmp.per),
                    ("PBV", cmp.pbv),
                )
                if v is not None
            )
            lines.append(
                f"- {cmp.symbol} ({cmp.cls}) skor {cmp.score:g}, {status}, return 1 tahun {cmp.return1y:g}%, volatilitas {cmp.volatility:g}%{extra}"
            )
    return "\n".join(lines) or "(tidak ada)"


def news_passages(items: list[NewsCtx]) -> list[Passage]:
    """Articles as passages (page 0 marks 'no page'), so numbering, citations and the guardrail treat them like documents."""
    return [
        Passage(
            chunk_id=-i,
            title=n.title,
            source_type="news",
            page=0,
            section=n.source or None,
            content=f"{n.source} {n.publishedAt[:10]}\n{n.summary or n.title}".strip(),
            key=n.url or n.title,
            ticker=None,
        )
        for i, n in enumerate(items, 1)
    ]


def passages_text(passages: list[Passage]) -> str:
    if not passages:
        return "(tidak ada kutipan relevan)"
    return "\n\n".join(
        f"[{i}] {'Berita: ' if p.source_type == 'news' else ''}{p.title}"
        + (f", hal. {p.page}" if p.page else "")
        + f"\n{p.content[:1800]}"
        for i, p in enumerate(passages, 1)
    )
