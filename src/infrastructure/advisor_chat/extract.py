import re
from dataclasses import dataclass

import pdfplumber

_NUM = re.compile(r"^\(?-?[\d.,]+\)?%?$")


@dataclass(frozen=True)
class Page:
    number: int
    text: str
    has_table: bool


def _tabular(text: str) -> bool:
    """Cheap table hint: several lines carry three or more numeric tokens."""
    rows = 0
    for line in text.splitlines():
        if sum(1 for t in line.split() if _NUM.match(t) and any(c.isdigit() for c in t)) >= 3:
            rows += 1
            if rows >= 3:
                return True
    return False


def extract_pages(path: str) -> tuple[list[Page], bool]:
    """Text per page. Returns the pages and whether the document looks scanned (almost no text)."""
    pages: list[Page] = []
    with pdfplumber.open(path) as pdf:
        for i, p in enumerate(pdf.pages, start=1):
            text = (p.extract_text() or "").replace("\x00", "").strip()  # Postgres rejects NUL
            pages.append(Page(i, text, _tabular(text)))
    empty = sum(1 for p in pages if len(p.text) < 30)
    scanned = bool(pages) and empty / len(pages) > 0.6
    return pages, scanned
