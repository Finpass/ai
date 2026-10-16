from pydantic import BaseModel, ConfigDict, Field


class TextStatsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(max_length=100_000)


class TextStatsResponse(BaseModel):
    words: int
    chars: int
