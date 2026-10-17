import re
from decimal import Decimal, InvalidOperation

_TICKER = re.compile(r"\b[A-Z]{4}\b")
_WORDS = re.compile(
    r"\b(buy|buying|bought|sell|selling|sold|beli|membeli|dibeli|jual|menjual|dijual|"
    r"bitcoin|btc|ethereum|eth|dogecoin|solana)\b|\$[A-Za-z]{2,5}\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_GROUPED = re.compile(r"^\d{1,3}(?:[.,]\d{3})+$")


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


def violations(text: str, allowed: set[Decimal]) -> list[str]:
    bad: list[str] = []
    if _WORDS.search(text) or _TICKER.search(text):
        bad.append("recommendation wording")
    for tok in _NUMBER.findall(text):
        if not _values(tok) & allowed:
            bad.append(f"untraceable number {tok}")
    return bad
