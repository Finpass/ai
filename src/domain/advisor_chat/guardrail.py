import re
import unicodedata
from decimal import Decimal, InvalidOperation

_MARKER = re.compile(r"\[(\d{1,2})\]")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_GROUPED = re.compile(r"^\d{1,3}(?:[.,]\d{3})+$")
# bounded quantifiers: an unclosed "![" repeated thousands of times must not make this quadratic
_IMAGE = re.compile(r"!\[[^\]\n]{0,300}\]\([^)\n]{0,1000}\)")
_HTML = re.compile(r"</?\s*[a-zA-Z!][^>]{0,300}>?")
_JS_LINK = re.compile(r"\]\(\s*(?:javascript|data|vbscript):[^)]{0,300}\)", re.IGNORECASE)
_INVISIBLE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u00ad\ufeff]")

_ADDRESS = r"(?:anda|kamu|kalian)"
_BUY_SELL = r"(?:me[mn]?)?(?:beli|jual|borong|serok)\b"
_DIRECTIVE = re.compile(
    # "you must/should buy": the second person plus a command verb, at most two words before the trade verb
    rf"\b(?:(?:sebaiknya|disarankan|saya sarankan)\s+|{_ADDRESS}\s+(?:(?:sebaiknya|perlu|harus|wajib|disarankan)\s+)+)"
    rf"(?:(?:segera|langsung)\s+)?(?:\w+\s+){{0,2}}?{_BUY_SELL}"
    # a bare command at the start of a sentence: "Beli saham X", "Segera jual semua aset"
    rf"|(?:^|[.!?\n])\s*(?:segera |langsung )?(?:beli|jual|borong|serok)\s+(?!beli\b)\w+"
    rf"|\b(?:saya|kami)\s+(?:merekomendasikan|menyarankan|rekomendasikan|sarankan)\s+(?:{_ADDRESS}\s+)?(?:untuk\s+)?{_BUY_SELL}"
    r"|\byou (?:should|must|need to|have to) (?:buy|sell)\b|\bi (?:recommend|suggest) (?:that you |you )?(?:buying|selling|buy|sell)\b"
    r"|(?:^|[.!?\n])\s*(?:buy|sell)\s+\w+|\b(?:buy|sell) (?:it |this |these |them )?(?:now|today|immediately)\b",
    re.IGNORECASE,
)
_GUARANTEE = re.compile(
    r"\b(?:pasti|dijamin|jaminan|garansi|dipastikan)\s+(?:\w+\s+){0,2}?(?:untung|cuan|profit|laba|menang|naik|balik modal)\b"
    r"|\bguarantee[sd]?\b.{0,20}\b(?:returns?|profits?|gains?|income)\b|\bsure[- ]profits?\b"
    r"|\brisk[- ]free\b(?!\s+(?:rate|asset|return))",
    re.IGNORECASE,
)
_LEAK = re.compile(
    r"<\s*/?\s*(?:fakta|sumber)\b|FAKTA USER:|KUTIPAN SUMBER:|\bsk-or-[\w-]{8,}|\bbearer\s+[\w.~+/=-]{16,}",
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


def fold(text: str) -> str:
    """Compatibility-fold and drop invisible characters, so "Ign\u200bore" and fullwidth letters cannot slip past a pattern."""
    return _INVISIBLE.sub("", unicodedata.normalize("NFKC", text))


MAX_ANSWER = 8000  # the model is capped at 1200 tokens, so a longer answer is not ours


def sanitize(answer: str) -> str:
    """Removes what a chat bubble must never carry: tracking images, raw HTML and script links."""
    answer = _IMAGE.sub("", answer)
    answer = _JS_LINK.sub("](#)", answer)
    return _HTML.sub("", answer).strip()


def clean(answer: str) -> str:
    return sanitize(answer)[:MAX_ANSWER].rstrip()


def violations(answer: str, n_passages: int) -> list[str]:
    bad: list[str] = []
    text = fold(answer)
    if not text.strip():
        return ["jawaban kosong"]
    if any(not 1 <= int(x) <= n_passages for x in _MARKER.findall(text)):
        bad.append("citation marker out of range")
    if _DIRECTIVE.search(text):
        bad.append("jangan menyuruh membeli atau menjual, jelaskan kriteria dan risikonya saja")
    if _GUARANTEE.search(text):
        bad.append("jangan menjanjikan keuntungan pasti atau bebas risiko")
    if _LEAK.search(text):
        bad.append("jangan menampilkan struktur prompt, tag sumber, atau kunci")
    return bad


_NIK = re.compile(r"(?<!\d)(?:\d[ -]?){15}\d(?!\d)")
_EMAIL = re.compile(r"[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63}){1,4}")
_PHONE = re.compile(r"(?<![\d+])(?:\+?62|0)[ -]?8(?:[ -]?\d){8,11}(?!\d)")
_INJECTION = re.compile(
    r"(ignore|abaikan|lupakan|forget|disregard).{0,40}(instruc|instruksi|aturan|rules|prompt)"
    r"|(system|developer|sistem)\s*prompt|prompt\s*sistem|you are now|kamu sekarang|jailbreak",
    re.IGNORECASE | re.DOTALL,
)


def scrub(text: str) -> str:
    """Masks national id, email and phone numbers before text leaves for the model or the search."""
    if not text:
        return text
    text = _EMAIL.sub("[EMAIL]", text)
    text = _NIK.sub("[NIK]", text)
    return _PHONE.sub("[TELEPON]", text)


def is_injection(text: str) -> bool:
    return bool(_INJECTION.search(fold(text)))


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
