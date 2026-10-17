import re
from decimal import Decimal, InvalidOperation

_ADVICE = re.compile(
    r"\b(sebaiknya|disarankan|saya sarankan|anda harus|kamu harus|you should|i recommend|i suggest|we recommend)\b"
    r"|\b(beli|buy|jual|sell)\s+(saham|aset|sekarang|now|it|this|ini)\b"
    r"|pasti\s+(naik|untung|cuan)|dijamin|guaranteed|risk[- ]free",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![\w\[])\d[\d.,]*%?")
_MARKER = re.compile(r"\[(\d{1,2})\]")


def _values(token: str) -> set[Decimal]:
    """Every plausible reading of a number token: '1.234' can be one thousand or 1.234."""
    t = token.rstrip("%").rstrip(".,")
    out: set[Decimal] = set()
    cands = {
        t,
        t.replace(".", "").replace(",", ""),
        t.replace(".", "").replace(",", "."),
        t.replace(",", ""),
    }
    for c in cands:
        try:
            out.add(Decimal(c))
        except InvalidOperation:
            continue
    return out


def numbers_in(text: str) -> set[Decimal]:
    vals: set[Decimal] = set()
    for tok in _NUMBER.findall(text):
        vals |= _values(tok)
    return vals


def violations(answer: str, allowed_text: str, n_passages: int, needs_citation: bool) -> list[str]:
    found: list[str] = []
    if m := _ADVICE.search(answer):
        found.append(f"advice wording: {m.group(0)}")
    allowed = numbers_in(allowed_text)
    for tok in _NUMBER.findall(answer):
        vals = _values(tok)
        if not vals:
            continue
        # small whole numbers (counts, list items) are harmless; everything else must come from the context
        if any(v == v.to_integral_value() and v < 100 for v in vals):
            continue
        if not (vals & allowed):
            found.append(f"number not in context: {tok}")
    marks = {int(x) for x in _MARKER.findall(answer)}
    if any(not 1 <= x <= n_passages for x in marks):
        found.append("citation marker out of range")
    if needs_citation and n_passages and not marks:
        found.append("missing citation")
    return found[:6]


_GAP = re.compile(
    r"tidak (?:ditemukan|menemukan|ada informasi|tersedia)|belum (?:ditemukan|menemukan)"
    r"|(?:not|no) (?:found|information|relevant)|could not find|cannot find|isn'?t (?:covered|in)",
    re.IGNORECASE,
)


def admits_gap(answer: str) -> bool:
    """The model says the sources do not cover the question: a legitimate answer without citations."""
    return bool(_GAP.search(answer))


def used_markers(answer: str, n_passages: int) -> list[int]:
    return sorted({int(x) for x in _MARKER.findall(answer) if 1 <= int(x) <= n_passages})
