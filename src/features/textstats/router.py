from fastapi import APIRouter

from src.core.envelope import Success
from src.core.security import require_internal_key
from src.features.textstats import service
from src.features.textstats.schemas import TextStatsRequest, TextStatsResponse

router = APIRouter(
    prefix="/ml/text-stats", tags=["text-stats"], dependencies=[require_internal_key]
)


@router.post("", response_model=Success[TextStatsResponse])
async def text_stats(req: TextStatsRequest) -> Success[TextStatsResponse]:
    return Success(message="Text analyzed", data=service.analyze(req.text))
