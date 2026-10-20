import re
from decimal import Decimal, InvalidOperation

_MARKER = re.compile(r"\[(\d{1,2})\]")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_GROUPED = re.compile(r"^\d{1,3}(?:[.,]\d{3})+$")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_DIRECTIVE = re.compile(
    r"\b(?:sebaiknya|disarankan|saya sarankan|anda (?:perlu|harus)|kamu (?:perlu|harus))\b"
    r".{0,30}\b(?:beli|membeli|jual|menjual)\b"
    r"|\byou should (?:buy|sell)\b|\bi recommend (?:buying|selling)\b",
    re.IGNORECASE,
)
_GAP = re.compile(
    r"tidak (?:ditemukan|menemukan|ada informasi|tersedia)|belum (?:ditemukan|menemukan)"
    r"|(?:not|no) (?:found|information|relevant)|could not find|cannot find|isn'?t (?:covered|in)",
    re.IGNORECASE,
)


def _values(token: str) -> set[Decimal]:
    out: set[Decimal] = set()
    try:
        if _GROUPED.match(token):
            out.add(Decimal(re.sub(r"[.,]", "", token)))
        if re.fullmatch(r"\d+[.,]\d+", token):
            out.add(Decimal(token.replace(",", ".")))
        elif not _GROUPED.match(token):
            out.add(Decimal(re.sub(r"[.,]", "", token)))
    except InvalidOperation:
        pass
    return {v.normalize() for v in out}


def sanitize(answer: str) -> str:
    return _IMAGE.sub("", answer).strip()


def violations(answer: str, n_passages: int) -> list[str]:
    bad: list[str] = []
    if any(not 1 <= int(x) <= n_passages for x in _MARKER.findall(answer)):
        bad.append("citation marker out of range")
    if _DIRECTIVE.search(answer):
        bad.append("jangan menyuruh membeli atau menjual, jelaskan kriteria dan risikonya saja")
    return bad


def untraceable_numbers(answer: str, source: str) -> list[str]:
    known: set[Decimal] = set()
    for tok in _NUMBER.findall(source):
        known |= _values(tok)
    bad: list[str] = []
    for tok in _NUMBER.findall(_MARKER.sub("", answer)):
        if tok.isdigit() and int(tok) <= 10:
            continue
        if not _values(tok) & known:
            bad.append(f"untraceable number {tok}")
    return bad


def admits_gap(answer: str) -> bool:
    return bool(_GAP.search(answer))


def used_markers(answer: str, n_passages: int) -> list[int]:
    return sorted({int(x) for x in _MARKER.findall(answer) if 1 <= int(x) <= n_passages})
