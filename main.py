import uvicorn

from src.core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",  # noqa: S104
        port=settings.port,
        reload=not settings.is_production,
    )
