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
src/app.py                 composition: middleware, error handlers, routers
src/core/                  config, errors (AppError + codes), envelope, request context, security
src/domain/<slice>/        pure rules, no FastAPI/IO
src/features/<slice>/      router.py (HTTP), schemas.py (DTOs), service.py (use case)
src/infrastructure/        adapters: LLM clients, vector stores, external APIs
```

Rules:
- A slice owns its router, schemas and service. Slices do not import each other.
- Raise `AppError` (`core/errors.py`) from domain/service code; never `HTTPException`.
- Responses use `Success[T]` from `core/envelope.py`. The error shape is produced only by `core/error_handler.py`.
- `features/textstats` is a working template; copy it, then delete it when the first real slice lands.

Contract (shared with backend and frontend): `../backend/docs/error-contract.md`.
