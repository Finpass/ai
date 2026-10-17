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


class HoldingCtx(BaseModel):
    symbol: str
    name: str = ""
    assetClass: str = ""
    weightPct: float = 0


class RecAsset(BaseModel):
    ticker: str
    weightPct: float = 0


class RecClass(BaseModel):
    cls: str = Field(alias="class")
    allocationPct: float = 0
    assets: list[RecAsset] = []

    model_config = {"populate_by_name": True}


class UserContext(BaseModel):
    profile: ProfileCtx | None = None
    holdings: list[HoldingCtx] = []
    recommended: list[RecClass] = []


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    lang: Literal["id", "en"] = "id"
    history: list[Turn] = Field(default=[], max_length=12)
    userContext: UserContext = UserContext()
    runId: str | None = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


class Citation(BaseModel):
    n: int
    title: str
    page: int
    sourceType: str
    key: str


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
