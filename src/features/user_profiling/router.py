from typing import Annotated

from fastapi import APIRouter, Depends

from src.core.config import settings
from src.core.envelope import Success
from src.core.security import require_internal_key
from src.domain.user_profiling.schemas import ExplainRequest, ExplainResponse
from src.features.user_profiling import service
from src.infrastructure.openrouter import OpenRouterClient

router = APIRouter(prefix="/ml/profile", tags=["profile"], dependencies=[require_internal_key])


def get_llm() -> OpenRouterClient | None:
    return OpenRouterClient() if settings.openrouter_api_key else None


@router.post("/explain", response_model=Success[ExplainResponse])
async def explain(
    req: ExplainRequest, llm: Annotated[OpenRouterClient | None, Depends(get_llm)]
) -> Success[ExplainResponse]:
    return Success(message="Explanation generated", data=await service.explain(req, llm))
