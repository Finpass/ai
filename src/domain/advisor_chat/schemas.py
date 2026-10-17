from typing import Literal

from pydantic import BaseModel, Field


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ProfileCtx(BaseModel):
    riskProfile: str
    ladderLevel: int
    sharia: str = "no"
    horizonYears: float = 0
    goal: str | None = None
    goalAmountIDR: int | None = None
    incomeIDR: int | None = None
    expensesIDR: int | None = None
    surplusIDR: int | None = None
    installmentIDR: int | None = None
    savingsIDR: int | None = None
    investableIDR: int | None = None
    dependents: int | None = None
    employment: str | None = None
    knowledge: str | None = None
    emergencyMonths: float | None = None
    emergencyTargetMonths: float | None = None
    healthCategory: str | None = None
    healthScore: float | None = None
    heldKinds: list[str] = []
    priorities: list[str] = []


class HoldingCtx(BaseModel):
    symbol: str
    name: str = ""
    assetClass: str = ""
    weightPct: float = 0
    valueIDR: int = 0


class RecAsset(BaseModel):
    ticker: str
    weightPct: float = 0


class RecClass(BaseModel):
    cls: str = Field(alias="class")
    allocationPct: float = 0
    assets: list[RecAsset] = []

    model_config = {"populate_by_name": True}


class CompareCtx(BaseModel):
    """Rule outcome for one product the user asked to compare; the model explains it, never changes it."""

    symbol: str
    name: str = ""
    cls: str = Field(default="", alias="class")
    eligible: bool = True
    gate: str = ""
    score: float = 0
    rank: int = 0
    return1y: float = 0
    volatility: float = 0
    sharpe: float | None = None
    divYield: float | None = None
    per: float | None = None
    pbv: float | None = None

    model_config = {"populate_by_name": True}


class NewsCtx(BaseModel):
    """A recent market article the backend picked; the advisor may cite it when documents fall short."""

    title: str
    source: str = ""
    url: str = ""
    publishedAt: str = ""
    summary: str = Field(default="", max_length=1200)


class UserContext(BaseModel):
    profile: ProfileCtx | None = None
    holdings: list[HoldingCtx] = []
    recommended: list[RecClass] = []
    comparison: list[CompareCtx] = Field(default=[], max_length=4)
    news: list[NewsCtx] = Field(default=[], max_length=8)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    lang: Literal["id", "en"] = "id"
    history: list[Turn] = Field(default=[], max_length=12)
    userContext: UserContext = UserContext()
    runId: str | None = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


class Citation(BaseModel):
    n: int
    title: str
    page: int | None = None
    sourceType: str
    key: str
    url: str | None = None


class Trace(BaseModel):
    intent: str
    tickers: list[str]
    chunkIds: list[int]
    topScore: float


class Turn_(BaseModel):
    turn_id: str
    sequence: int
    title: str
    type: str
    status: str
    timestamp: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool
    source: Literal["llm", "fallback"]
    trace: Trace
    turns: list[Turn_] = []
