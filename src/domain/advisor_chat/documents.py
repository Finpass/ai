from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    number: int
    text: str
    has_table: bool


@dataclass
class Passage:
    id: int | str
    title: str
    source_type: str
    page_start: int
    section: str
    content: str
    r2_key: str
    ticker: str | None = None
    score: float = 0.0
    tier: int = 1
    nomor: str = ""
    publisher: str = "OJK"
    tanggal: int = 0
