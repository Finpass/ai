from dataclasses import dataclass

from src.core.errors import unprocessable


@dataclass(frozen=True)
class TextStats:
    words: int
    chars: int


def compute(text: str) -> TextStats:
    stripped = text.strip()
    if not stripped:
        raise unprocessable("text must not be blank")
    return TextStats(words=len(stripped.split()), chars=len(stripped))
