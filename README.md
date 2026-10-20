# ai (FastAPI)

Python 3.13, FastAPI, pydantic v2, uv, ruff, mypy (strict), pytest.

```
cp .env.example .env
uv sync
make run      # :8001, docs at /docs (dev only)
make check    # ruff + mypy + pytest
```

All routes except `/health` require `X-Internal-Api-Key`; only the Go backend calls this service.

## Architecture: vertical slices + DDD

```
src/app.py                      composition: middleware, error handlers, routers
src/core/                       config, errors (AppError + codes), envelope, request context, security
src/domain/<feature>/           pure logic and Pydantic schemas: no FastAPI, DB or LLM
src/features/<feature>/         router.py (HTTP only) + service.py (use case)
src/infrastructure/             adapters: DB, R2, LLM, vector store; prompt/*.txt holds every prompt
```

Slices: `advisor_chat` (RAG advisor), `user_profiling` (profile explanation). `wa_brief` is planned.
Imports go one way: router -> service -> domain / infra. `domain` imports nothing from above.

Rules:
- A slice owns its router, schemas and service. Slices do not import each other.
- Raise `AppError` (`core/errors.py`) from domain/service code; never `HTTPException`.
- Responses use `Success[T]` from `core/envelope.py`. The error shape is produced only by `core/error_handler.py`.

Contract (shared with backend and frontend): `../backend/docs/error-contract.md`.

## Tracing

Traces, token counts and cost for every chat turn and profile explanation can be sent to Arize Phoenix. Setup, the span tree and privacy settings: [docs/TRACING.md](docs/TRACING.md).
