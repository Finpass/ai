import logging
from typing import Any

import httpx2 as httpx

from src.core.config import settings
from src.infrastructure.observability import tracing

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
        purpose: str = "chat_completion",
    ) -> str:
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "X-Title": "Finpass",
        }
        if settings.openrouter_app_url:
            headers["HTTP-Referer"] = settings.openrouter_app_url
        provider: dict[str, Any] = {
            "allow_fallbacks": settings.openrouter_allow_fallbacks,
            "data_collection": settings.openrouter_data_collection,
        }
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
            "usage": {"include": True},  # OpenRouter then reports tokens and the exact cost in USD
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        url = settings.openrouter_base_url.rstrip("/") + "/chat/completions"
        limit = timeout_seconds or settings.openrouter_timeout_seconds
        shown = ("temperature", "max_tokens", "provider", "response_format")
        params = {k: body[k] for k in shown if k in body}
        with tracing.llm_span(
            purpose,
            model=body["model"],
            messages=messages,
            params=params,
            purpose=purpose,
        ) as t:
            try:
                async with httpx.AsyncClient(transport=self._transport, timeout=limit) as http:
                    resp = await http.post(url, json=body, headers=headers)
                resp.raise_for_status()
                data: Any = resp.json()
                content = data["choices"][0]["message"]["content"]
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                logger.warning("openrouter call failed: %s", type(exc).__name__)
                err = LLMError(type(exc).__name__)
                t.fail(err)
                raise err from exc
            if not isinstance(content, str):
                logger.warning("openrouter returned no text content")
                err = LLMError("non-text content")
                t.fail(err)
                raise err
            tracing.llm_result(t, content, data)
            return content
