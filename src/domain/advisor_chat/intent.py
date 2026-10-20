import re

REGULASI = re.compile(
    r"\b(ojk|pojk|padk|peraturan|aturan|regulasi|undang|hukum|sanksi|perizinan|izin|lps|ketentuan)\b",
    re.I,
)
PERUSAHAAN = re.compile(
    r"\b(laporan keuangan|lapkeu|financial statement|laba|rugi|pendapatan|revenue|ekuitas|aset|liabilitas|utang|dividen|emiten|saham)\b",
    re.I,
)
PORTOFOLIO = re.compile(
    r"\b(portofolio|portfolio|holding|alokasi|allocation"
    r"|rekomendasi (?:portofolio|produk|aset|saya)|recommended (?:portfolio|allocation)"
    r"|profil saya|my profile|my portfolio)\b",
    re.I,
)
TICKER = re.compile(r"\b([A-Z]{4})(?:\.JK)?\b")
STOP = {"OJKK", "YANG", "APAKAH"}

IDX = ["idx_financial_statement", "idx_laporan_keuangan"]
OJK = ["ojk_regulasi", "ojk_edukasi"]


def detect_intent(message: str, has_ticker: bool) -> str:
    if has_ticker or (PERUSAHAAN.search(message) and not REGULASI.search(message)):
        return "perusahaan"
    if PORTOFOLIO.search(message):
        return "portofolio"
    if REGULASI.search(message):
        return "regulasi"
    return "edukasi"


def source_types(intent: str, has_ticker: bool) -> list[str] | None:
    if intent == "perusahaan":
        return IDX
    if intent == "regulasi":
        return ["ojk_regulasi", "ojk_edukasi"]
    if intent == "edukasi":
        return ["ojk_edukasi", "ojk_regulasi"]
    return None  # portofolio: any source


def mentioned_tickers(message: str, known: set[str]) -> list[str]:
    """Uppercase 4-letter symbols the user typed that exist in the corpus (BBCA or BBCA.JK)."""
    found = [m for m in TICKER.findall(message) if m in known and m not in STOP]
    return list(dict.fromkeys(found))


def held_tickers(symbols: list[str], known: set[str]) -> list[str]:
    base = [s.split(".")[0].upper() for s in symbols]
    return [b for b in dict.fromkeys(base) if b in known]
