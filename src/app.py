import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from phoenix.otel import register

from src.core.config import settings
from src.core.envelope import Success
from src.core.error_handler import add_error_handlers
from src.core.request_context import RequestContextMiddleware
from src.features.advisor_chat.router import router as advisor_chat_router
from src.features.user_profiling.router import router as user_profiling_router

logging.basicConfig(
    level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

TRACING_ENABLED = bool(os.getenv("PHOENIX_COLLECTOR_ENDPOINT"))

if TRACING_ENABLED:
    register(project_name="ai-advisor-service", auto_instrument=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.rag_enabled:
        from src.infrastructure.advisor_chat import embedder, loader

        await asyncio.to_thread(loader.ensure_corpus)

        asyncio.get_running_loop().run_in_executor(None, embedder.warmup)
    yield


app = FastAPI(
    title="Shenzen AI", docs_url=None if settings.is_production else "/docs", lifespan=lifespan
)

if TRACING_ENABLED:
    FastAPIInstrumentor.instrument_app(app, excluded_urls="health")

app.add_middleware(RequestContextMiddleware)
add_error_handlers(app)

app.include_router(user_profiling_router)
app.include_router(advisor_chat_router)


@app.get("/health", tags=["health"], response_model=Success[None])
async def health() -> Success[None]:
    return Success(data=None)
