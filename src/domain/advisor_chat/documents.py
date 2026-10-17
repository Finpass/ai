from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    number: int
    text: str
    has_table: bool


@dataclass
class Passage:
    chunk_id: int
    title: str
    source_type: str
    page: int
    section: str | None
    content: str
    key: str
    ticker: str | None
    score: float = 0.0
