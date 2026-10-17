from typing import Annotated

from fastapi import APIRouter, Depends

from src.core.config import settings
from src.core.envelope import Success
from src.core.security import require_internal_key
from src.features.rag import service
from src.features.rag.schemas import ChatRequest, ChatResponse
from src.infrastructure.observability.turns import TurnEvent, turn_buffer
from src.infrastructure.openrouter import OpenRouterClient

router = APIRouter(prefix="/ml/rag", tags=["rag"], dependencies=[require_internal_key])


def get_llm() -> OpenRouterClient | None:
    return OpenRouterClient() if settings.openrouter_api_key else None


@router.post("/chat", response_model=Success[ChatResponse])
async def chat(
    req: ChatRequest, llm: Annotated[OpenRouterClient | None, Depends(get_llm)]
) -> Success[ChatResponse]:
    return Success(data=await service.chat(req, llm))


@router.get("/runs/{run_id}/turns", response_model=Success[list[TurnEvent]])
async def turns(run_id: str, since: int = 0) -> Success[list[TurnEvent]]:
    """Turn titles emitted so far for one chat run; the backend polls this while a reply is in flight."""
    return Success(data=turn_buffer.read(run_id, since))
