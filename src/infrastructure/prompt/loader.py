from functools import cache
from pathlib import Path

_DIR = Path(__file__).parent


@cache
def load(name: str) -> str:
    """Prompt text lives in <name>.txt beside this file, so wording changes never touch code."""
    return (_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()
