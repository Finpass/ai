from src.domain.textstats import stats
from src.features.textstats.schemas import TextStatsResponse


def analyze(text: str) -> TextStatsResponse:
    result = stats.compute(text)
    return TextStatsResponse(words=result.words, chars=result.chars)
