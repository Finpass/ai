import logging

from fastapi import FastAPI

from src.core.config import settings
from src.core.envelope import Success
from src.core.error_handler import add_error_handlers
from src.core.request_context import RequestContextMiddleware
from src.features.textstats.router import router as textstats_router

logging.basicConfig(
    level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

app = FastAPI(title="Shenzen AI", docs_url=None if settings.is_production else "/docs")
app.add_middleware(RequestContextMiddleware)
add_error_handlers(app)

app.include_router(textstats_router)


@app.get("/health", tags=["health"], response_model=Success[None])
async def health() -> Success[None]:
    return Success(data=None)
