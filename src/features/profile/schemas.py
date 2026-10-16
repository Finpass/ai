from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Persona(BaseModel):
    model_config = ConfigDict(extra="ignore")

    firstName: str | None = Field(default=None, max_length=100)


class Flag(BaseModel):
    model_config = ConfigDict(extra="ignore")

    code: str


class TargetMonths(BaseModel):
    model_config = ConfigDict(extra="ignore")

    min: float = Field(ge=0)
    max: float = Field(ge=0)


class FinancialHealth(BaseModel):
    model_config = ConfigDict(extra="ignore")

    financialSecurity: float = Field(ge=0, le=100)
    financialSafety: float = Field(ge=0, le=100)
    financialGrowth: float = Field(ge=0, le=100)
    financialWisdom: float = Field(ge=0, le=100)
    financialFreedom: float = Field(ge=0, le=100)
    totalScore: float = Field(ge=0, le=100)
    category: Literal["Really Struggling", "Struggling", "Coping", "Thriving"]


class Metrics(BaseModel):
    model_config = ConfigDict(extra="ignore")

    monthlyIncome: float = Field(ge=0)
    monthlyExpenses: float = Field(ge=0)
    monthlyInstallment: float = Field(default=0, ge=0)
    monthlySurplus: float
    savingsRate: float = 0
    dti: float = 0
    investableRatio: float = 0


class ProfileResult(BaseModel):
    """Rule-engine Result without explanation; unknown fields are ignored."""

    model_config = ConfigDict(extra="ignore")

    capacityScore: float | None = Field(default=None, ge=0, le=100)
    willingnessScore: float | None = Field(default=None, ge=0, le=100)
    finalProfile: Literal["Conservative", "Moderate", "Aggressive"]
    profileLabel: Literal["provisional", "final"]
    riskScoreEquivalent: float | None = None
    ladderLevel: int = Field(ge=1, le=5)
    ladderProgress: float = Field(ge=0, le=100)
    emergencyFundMonths: float = Field(ge=0)
    emergencyTargetMonths: TargetMonths
    flags: list[Flag] = []
    financialHealth: FinancialHealth
    metrics: Metrics


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    locale: Literal["id", "en"] = "id"
    persona: Persona = Persona()
    result: ProfileResult
    answers: dict[str, Any] = {}


class ExplainResponse(BaseModel):
    locale: Literal["id", "en"]
    source: Literal["llm", "template"]
    model: str | None
    summary: str
    risk: str
    ladder: str
    health: str
    nextStep: str
    disclaimer: str
