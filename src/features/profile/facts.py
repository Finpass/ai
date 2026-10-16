import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from src.features.profile.schemas import ExplainRequest

DIMENSIONS = ("Security", "Safety", "Growth", "Wisdom", "Freedom")
# Fixed numbers any text may mention: score scale and ladder size.
CONSTANTS = (100, 5)


@dataclass(frozen=True)
class Facts:
    locale: str
    name: str
    profile: str
    provisional: bool
    capacity: float | None
    willingness: float | None
    risk_score: float | None
    level: int
    progress: float
    months: float
    target_min: float
    target_max: float
    income: float
    expenses: float
    installment: float
    surplus: float
    savings_rate_pct: float
    dti_pct: float
    savings: float
    target_amount: float
    gap: float
    dims: dict[str, float]
    total: float
    category: str
    strongest: str
    weakest: str
    flags: tuple[str, ...]


def _first_name(raw: str | None) -> str:
    parts = re.findall(r"[^\W\d_]+(?:['-][^\W\d_]+)*", raw or "")
    return parts[0][:30] if parts else ""


def build(req: ExplainRequest) -> Facts:
    r, m, h = req.result, req.result.metrics, req.result.financialHealth
    dims = {
        "Security": h.financialSecurity,
        "Safety": h.financialSafety,
        "Growth": h.financialGrowth,
        "Wisdom": h.financialWisdom,
        "Freedom": h.financialFreedom,
    }
    raw = req.answers.get("q4")
    if isinstance(raw, int | float) and not isinstance(raw, bool) and raw >= 0:
        savings = float(raw)
    else:
        savings = r.emergencyFundMonths * m.monthlyExpenses
    target_amount = r.emergencyTargetMonths.min * m.monthlyExpenses
    return Facts(
        locale=req.locale,
        name=_first_name(req.persona.firstName),
        profile=r.finalProfile,
        provisional=r.profileLabel == "provisional",
        capacity=r.capacityScore,
        willingness=r.willingnessScore,
        risk_score=r.riskScoreEquivalent,
        level=r.ladderLevel,
        progress=r.ladderProgress,
        months=r.emergencyFundMonths,
        target_min=r.emergencyTargetMonths.min,
        target_max=r.emergencyTargetMonths.max,
        income=m.monthlyIncome,
        expenses=m.monthlyExpenses,
        installment=m.monthlyInstallment,
        surplus=m.monthlySurplus,
        savings_rate_pct=m.savingsRate * 100,
        dti_pct=m.dti * 100,
        savings=savings,
        target_amount=target_amount,
        gap=max(0.0, target_amount - savings),
        dims=dims,
        total=h.totalScore,
        category=h.category,
        strongest=max(dims, key=lambda k: dims[k]),
        weakest=min(dims, key=lambda k: dims[k]),
        flags=tuple(f.code for f in r.flags),
    )


def numbers(f: Facts) -> set[Decimal]:
    """Every figure a generated text may cite, with 0/1/2-decimal roundings."""
    raw: list[float | None] = [
        f.capacity, f.willingness, f.risk_score, f.level, f.progress, f.months,
        f.target_min, f.target_max, f.income, f.expenses, f.installment, f.surplus,
        f.savings_rate_pct, f.dti_pct, f.savings, f.target_amount, f.gap, f.total,
        *f.dims.values(), *CONSTANTS,
    ]  # fmt: skip
    out: set[Decimal] = set()
    for v in raw:
        if v is None:
            continue
        d = Decimal(str(v))
        for places in (1, 2):
            out.add(round(d, places).normalize())
        if abs(d) >= 100:  # whole rupiah/percent only; "2 bulan" for 1.5 stays untraceable
            out.add(Decimal(round(d)).normalize())
        out.add(d.normalize())
        out.add(abs(d).normalize())
    return out


def prompt_facts(f: Facts) -> dict[str, Any]:
    """Only rule-engine facts and the first name; no raw answers."""
    return {
        "firstName": f.name or None,
        "riskProfile": f.profile,
        "riskProfileIsProvisional": f.provisional,
        "capacityScore": f.capacity,
        "willingnessScore": f.willingness,
        "riskScoreEquivalent": f.risk_score,
        "financialLadderLevel": f.level,
        "ladderProgressPercent": f.progress,
        "emergencyFundMonths": f.months,
        "emergencyTargetMonths": f.target_min,
        "monthlyExpensesIDR": f.expenses,
        "monthlySurplusIDR": f.surplus,
        "monthlyInstallmentIDR": f.installment,
        "debtToIncomePercent": f.dti_pct,
        "savingsRatePercent": f.savings_rate_pct,
        "emergencyFundTargetIDR": f.target_amount,
        "emergencyFundGapIDR": f.gap,
        "healthScores": f.dims,
        "healthTotal": f.total,
        "healthCategory": f.category,
        "strongestDimension": f.strongest,
        "weakestDimension": f.weakest,
        "flags": list(f.flags),
    }
