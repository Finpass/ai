import re
from dataclasses import dataclass

from llama_index.core.node_parser import SentenceSplitter

from src.domain.advisor_chat.documents import Page

# BGE-M3 sees about 1.4 tokens per Indonesian word; sizes below are in words.
SIZES = {
    "ojk_regulasi": (320, 40),
    "ojk_edukasi": (330, 45),
    "idx_financial_statement": (330, 35),
    "idx_laporan_keuangan": (330, 35),
}
_HEADING = re.compile(r"^(?:pasal\s+\d+\w*|bab\s+[ivxlc\d]+\b.*|[A-Z][A-Z0-9 ,.&/()-]{6,80})$")


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    page_start: int
    page_end: int
    section: str | None
    has_table: bool
    content: str


def _words(text: str) -> list[str]:
    return text.split()


def _heading(text: str) -> str | None:
    for line in text.splitlines()[:6]:
        s = line.strip()
        if 6 <= len(s) <= 90 and _HEADING.match(s):
            return s
    return None


def chunk_pages(pages: list[Page], source_type: str, title: str) -> list[Chunk]:
    size, overlap = SIZES[source_type]
    splitter = SentenceSplitter(chunk_size=size, chunk_overlap=overlap, tokenizer=_words)
    # one flat text with page markers lets chunks span page borders and still report page ranges
    out: list[Chunk] = []
    section: str | None = None
    buf: list[tuple[int, str, bool]] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        text = "\n".join(t for _, t, _ in buf)
        start, end = buf[0][0], buf[-1][0]
        table = any(tb for _, _, tb in buf)
        for part in splitter.split_text(text):
            part = part.strip()
            if len(part) < 40:
                continue
            body = f"{title}\n{section}\n{part}" if section else f"{title}\n{part}"
            out.append(Chunk(len(out), start, end, section, table, body))
        buf = []

    budget = 0
    for pg in pages:
        if len(pg.text) < 30:
            continue
        h = _heading(pg.text)
        if h and h != section:
            flush()
            budget = 0
            section = h
        buf.append((pg.number, pg.text, pg.has_table))
        budget += len(pg.text.split())
        if budget >= size * 2:
            flush()
            budget = 0
    flush()
    return out
