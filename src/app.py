import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.core.config import settings
from src.core.envelope import Success
from src.core.error_handler import add_error_handlers
from src.core.request_context import RequestContextMiddleware
from src.features.advisor_chat.router import router as advisor_chat_router
from src.features.user_profiling.router import router as user_profiling_router
from src.infrastructure.observability import errors, tracing

logging.basicConfig(
    level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.rag_enabled:
        from src.infrastructure.advisor_chat import embedder, loader

        await asyncio.to_thread(loader.ensure_corpus)

        asyncio.get_running_loop().run_in_executor(None, embedder.warmup)
    yield


errors.setup_error_reporting()  # before the app exists, so Sentry can hook FastAPI

app = FastAPI(
    title="Shenzen AI",
    docs_url=None if settings.is_production else "/docs",
    lifespan=lifespan,
    # FastAPI 0.143 traces HTTP requests itself. Those spans have no input or output and the
    # turn-title polling floods Phoenix, so only the spans we open by hand are exported.
    telemetry={"tracing": False, "operation_spans": False, "metrics": False, "logs": False},
)

tracing.setup_tracing()  # Phoenix: the spans we open by hand (see tracing.py)

app.add_middleware(RequestContextMiddleware)
add_error_handlers(app)

app.include_router(user_profiling_router)
app.include_router(advisor_chat_router)


@app.get("/health", tags=["health"], response_model=Success[None])
async def health() -> Success[None]:
    return Success(data=None)
