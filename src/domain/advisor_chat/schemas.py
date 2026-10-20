from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Short = Annotated[str, StringConstraints(max_length=120)]


class _Strict(BaseModel):
    """Every model here refuses NaN and Infinity, which would otherwise be printed into the prompt."""

    model_config = ConfigDict(allow_inf_nan=False)


class Turn(_Strict):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class _NullAsMissing(_Strict):
    """Go encodes an empty slice it never initialised as null; treat null as 'not sent' so defaults apply."""

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data: Any) -> Any:
        return {k: v for k, v in data.items() if v is not None} if isinstance(data, dict) else data


class ProfileCtx(_NullAsMissing):
    riskProfile: Short
    ladderLevel: int
    sharia: Short = "no"
    horizonYears: float = 0
    goal: str | None = Field(default=None, max_length=2000)
    goalAmountIDR: int | None = None
    incomeIDR: int | None = None
    expensesIDR: int | None = None
    surplusIDR: int | None = None
    installmentIDR: int | None = None
    savingsIDR: int | None = None
    investableIDR: int | None = None
    dependents: int | None = None
    employment: Short | None = None
    knowledge: Short | None = None
    emergencyMonths: float | None = None
    emergencyTargetMonths: float | None = None
    healthCategory: Short | None = None
    healthScore: float | None = None
    heldKinds: list[Short] = Field(default=[], max_length=30)
    priorities: list[Short] = Field(default=[], max_length=20)

    # NEW: Validator untuk membatasi panjang teks bebas `goal` (Maks. 200 karakter)
    @field_validator("goal", mode="after")
    @classmethod
    def _trim_goal(cls, v: str | None) -> str | None:
        return v[:200] if v else v


class HoldingCtx(_Strict):
    symbol: Annotated[str, StringConstraints(max_length=24)]
    name: str = Field(default="", max_length=2000)
    assetClass: Short = ""
    weightPct: float = 0
    valueIDR: int = 0

    # NEW: Validator untuk membatasi panjang teks bebas `name` (Maks. 120 karakter)
    @field_validator("name", mode="after")
    @classmethod
    def _trim_name(cls, v: str) -> str:
        return v[:120]


class RecAsset(_Strict):
    ticker: Annotated[str, StringConstraints(max_length=24)]
    weightPct: float = 0


class RecClass(_Strict):
    cls: Short = Field(alias="class")
    allocationPct: float = 0
    assets: list[RecAsset] = Field(default=[], max_length=30)

    model_config = {"populate_by_name": True}


class CompareCtx(_Strict):
    """Rule outcome for one product the user asked to compare; the model explains it, never changes it."""

    symbol: Annotated[str, StringConstraints(max_length=24)]
    name: Short = ""
    cls: Short = Field(default="", alias="class")
    eligible: bool = True
    gate: Short = ""
    score: float = 0
    rank: int = 0
    return1y: float = 0
    volatility: float = 0
    sharpe: float | None = None
    divYield: float | None = None
    per: float | None = None
    pbv: float | None = None
    roe: float | None = None
    beta: float | None = None

    model_config = {"populate_by_name": True}


class NewsCtx(_Strict):
    """A recent market article the backend picked; the advisor may cite it when documents fall short."""

    title: str = Field(max_length=5000)
    source: str = Field(default="", max_length=500)
    url: str = Field(default="", max_length=2000)
    publishedAt: str = Field(default="", max_length=100)
    summary: str = Field(default="", max_length=20000)

    @field_validator("title", "summary", mode="after")
    @classmethod
    def _trim(cls, v: str) -> str:
        """Feed text has no length bound; cut it here so one long article never rejects the whole question."""
        return v[:1200]


class UserContext(_NullAsMissing):
    profile: ProfileCtx | None = None
    holdings: list[HoldingCtx] = Field(default=[], max_length=100)
    recommended: list[RecClass] = Field(default=[], max_length=12)
    comparison: list[CompareCtx] = Field(default=[], max_length=4)
    news: list[NewsCtx] = Field(default=[], max_length=8)


class ChatRequest(_Strict):
    message: str = Field(min_length=1, max_length=1000)
    lang: Literal["id", "en"] = "id"
    history: list[Turn] = Field(default=[], max_length=12)
    userContext: UserContext = UserContext()
    runId: str | None = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v


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
    chunkIds: list[str]
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
