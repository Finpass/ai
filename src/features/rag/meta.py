import re
from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class DocMeta:
    source_type: str
    title: str
    ticker: str | None
    fiscal_year: int | None
    language: str | None


_IDX = re.compile(r"^([A-Za-z0-9]{3,6})_")
_PERIOD = re.compile(r"-\d{4}-(Tahunan|I{1,3}|IV|TW\d)-", re.IGNORECASE)


def _clean(stem: str) -> str:
    stem = re.sub(r"^\d+_\d+_", "", stem)
    stem = re.sub(r"[_]+", " ", stem)
    stem = re.sub(r"-?ilovepdf-compressed|\.compressed", "", stem, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", stem).strip()


def parse_key(key: str) -> DocMeta | None:
    """Maps an R2 key to metadata, or None for keys outside the known corpus layout."""
    p = PurePosixPath(key)
    parts = p.parts
    if parts[0] == "ojk-regulasi":
        return DocMeta(
            "ojk_regulasi", _clean(p.stem), None, 2026 if "2026" in p.stem else None, "id"
        )
    if parts[0] == "ojk-edukasi":
        return DocMeta("ojk_edukasi", _clean(p.stem), None, None, None)
    if parts[0] == "idx-dokumen" and len(parts) >= 4:
        m = _IDX.match(p.name)
        if (
            not m
            or not parts[1].isdigit()
            or parts[2] not in ("Financial_Statement", "Laporan_Keuangan")
        ):
            return None
        english = parts[2] == "Financial_Statement"
        st = "idx_financial_statement" if english else "idx_laporan_keuangan"
        ticker = m.group(1).upper()
        year = int(parts[1])
        label = "Financial Statement" if english else "Laporan Keuangan"
        period = _PERIOD.search(p.name)
        suffix = f" {period.group(1)}" if period and period.group(1).lower() != "tahunan" else ""
        return DocMeta(
            st, f"{ticker} {label} {year}{suffix}", ticker, year, "en" if english else "id"
        )
    return None
