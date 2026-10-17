import logging
from typing import Any

import httpx2 as httpx

from src.core.config import settings

logger = logging.getLogger("openrouter")


class LLMError(Exception):
    """Any failure talking to the model; callers fall back instead of surfacing it."""


class OpenRouterClient:
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = True,
        max_tokens: int = 900,
        timeout_seconds: float | None = None,
        model: str | None = None,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "X-Title": "Finpass",
        }
        if settings.openrouter_app_url:
            headers["HTTP-Referer"] = settings.openrouter_app_url
        provider: dict[str, Any] = {"allow_fallbacks": settings.openrouter_allow_fallbacks}
        if settings.openrouter_provider_sort:
            provider["sort"] = settings.openrouter_provider_sort
        if order := [p.strip() for p in settings.openrouter_provider_order.split(",") if p.strip()]:
            provider["order"] = order
        body = {
            "model": model or settings.openrouter_model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "provider": provider,
            "reasoning": {"enabled": False},
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        url = settings.openrouter_base_url.rstrip("/") + "/chat/completions"
        limit = timeout_seconds or settings.openrouter_timeout_seconds
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=limit) as http:
                resp = await http.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data: Any = resp.json()
            content = data["choices"][0]["message"]["content"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            logger.warning("openrouter call failed: %s", type(exc).__name__)
            raise LLMError(type(exc).__name__) from exc
        if not isinstance(content, str):
            logger.warning("openrouter returned no text content")
            raise LLMError("non-text content")
        return content
