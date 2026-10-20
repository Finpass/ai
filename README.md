# ai (FastAPI)

Part of **Finpass**, a wealth advisory agent built for FinTechathon Shenzhen (Topic 3). Three repositories:

| Repository | What it is |
|------------|------------|
| [github.com/Finpass/frontend](https://github.com/Finpass/frontend) | Next.js web app: sign-in, profiling questionnaire, dashboard, advisor chat |
| [github.com/Finpass/backend](https://github.com/Finpass/backend) | Go API: auth, profile rule engine, portfolio, market data, WhatsApp daily brief |
| [github.com/Finpass/ai](https://github.com/Finpass/ai) | Python service: RAG advisor chat and profile wording (only the backend calls it) |

This is the **ai** repository.

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

Contract (shared with backend and frontend): [response and error contract](https://github.com/Finpass/backend#response-and-error-contract).

## Tracing

Traces, token counts and cost for every chat turn and profile explanation can be sent to Arize Phoenix. Setup, the span tree and privacy settings: [docs/TRACING.md](docs/TRACING.md).

## Tests

```
uv sync --extra rag --group dev
make test          # uv run pytest -q
make check         # ruff + mypy + pytest
```

The suite needs no network, no API key and no model download (the LLM, embeddings and Chroma are faked).

| File | Covers |
|------|--------|
| `tests/test_api.py` | envelope, error codes, internal API key, request ids |
| `tests/test_rag.py` | chat flow: retrieval, citations, guardrail, refusal and fallback paths |
| `tests/test_profile.py` | profile facts, number guardrail, template fallback, EN/ID wording |
| `tests/test_hardening.py` | prompt fencing, key rotation, provider settings |
| `tests/test_tracing.py` | Phoenix spans, token and cost totals, hidden content |
| `tests/test_config.py` | production refuses dev secrets |
| `tests/test_chroma_retrieval.py` | Chroma retrieval and reranking against a small index |

## Deployment (Railway, or any Docker host)

The service is one container: `Dockerfile` builds with uv (CPU-only torch), runs as a non-root user and starts
`uvicorn src.app:app` on `$PORT` (default 8001). `railway.json` sets the health check (`/health`).

1. Create a service from this repository; Railway builds the `Dockerfile`.
2. Set the variables below. Names and defaults are in `.env.example`; set real values only in the platform, never in git.
3. Keep it to **one worker**: ChromaDB's SQLite files do not support several processes.
4. First start downloads the vector index from Cloudflare R2 (`R2_*`), then loads the reranker model into memory
   (cached under `HF_HOME=/models`; mount a volume there to avoid downloading it on every deploy). Expect a slow
   first start and plan memory for a ~570M-parameter model plus the index.
5. Point the backend's `AI_SERVICE_URL` at this service and give both the same `INTERNAL_API_KEY`.

| Variable | Purpose |
|----------|---------|
| `APP_ENV=production` | production refuses the development API key and hides `/docs` |
| `INTERNAL_API_KEY` | shared secret the backend sends as `X-Internal-Api-Key` (`INTERNAL_API_KEY_PREVIOUS` during a rotation) |
| `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `RAG_MODEL` | LLM access and models |
| `R2_ENDPOINT`, `R2_BUCKET`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` | where the RAG index is downloaded from (use a read-only key) |
| `CHROMA_PATH` | local folder for the index |
| `PHOENIX_COLLECTOR_ENDPOINT`, `PHOENIX_API_KEY` | optional tracing; see [docs/TRACING.md](docs/TRACING.md) |
