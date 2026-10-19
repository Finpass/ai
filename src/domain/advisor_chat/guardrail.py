import re

_MARKER = re.compile(r"\[(\d{1,2})\]")


def violations(answer: str, n_passages: int) -> list[str]:
    """Deliberately loose: the answer may reason over the sources, so only dangling [n] markers are refused."""
    if any(not 1 <= int(x) <= n_passages for x in _MARKER.findall(answer)):
        return ["citation marker out of range"]
    return []


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
