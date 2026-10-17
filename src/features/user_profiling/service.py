import json
import logging
import re

from src.core.config import settings
from src.domain.user_profiling import facts as facts_mod
from src.domain.user_profiling import guardrail, template
from src.domain.user_profiling.schemas import ExplainRequest, ExplainResponse
from src.infrastructure.openrouter import LLMError, OpenRouterClient
from src.infrastructure.prompt.loader import load

logger = logging.getLogger("profile")

_KEYS = ("summary", "risk", "ladder", "health", "nextStep")
_MAX_LEN = 1200


def _parse(content: str) -> dict[str, str] | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    out: dict[str, str] = {}
    for k in _KEYS:
        v = data.get(k)
        if not isinstance(v, str) or not v.strip() or len(v) > _MAX_LEN:
            return None
        out[k] = v.strip()
    return out


async def _generate(f: facts_mod.Facts, llm: OpenRouterClient) -> dict[str, str] | None:
    lang = "Indonesian (Bahasa Indonesia)" if f.locale == "id" else "English"
    messages = [
        {"role": "system", "content": load("user_profiling_system").format(keys=", ".join(_KEYS))},
        {
            "role": "user",
            "content": f"Write in {lang}. Facts:\n"
            + json.dumps(facts_mod.prompt_facts(f), ensure_ascii=False),
        },
    ]
    try:
        content = await llm.complete(messages)
    except LLMError:
        return None
    parts = _parse(content)
    if parts is None:
        logger.warning("llm output rejected: invalid json")
        return None
    allowed = facts_mod.numbers(f)
    for key, text in parts.items():
        bad = guardrail.violations(text, allowed)
        if bad:
            logger.warning("llm output rejected: %s in %s", bad[0], key)
            return None
    return parts


async def explain(req: ExplainRequest, llm: OpenRouterClient | None) -> ExplainResponse:
    f = facts_mod.build(req)
    parts = await _generate(f, llm) if llm else None
    return ExplainResponse(
        locale=req.locale,
        source="llm" if parts else "template",
        model=settings.openrouter_model if parts else None,
        disclaimer=template.DISCLAIMER[req.locale],
        **(parts or template.render(f)),
    )
