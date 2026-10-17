import json
import logging
import re

from src.core.config import settings
from src.features.profile import facts as facts_mod
from src.features.profile import guardrail, template
from src.features.profile.schemas import ExplainRequest, ExplainResponse
from src.infrastructure.openrouter import LLMError, OpenRouterClient

logger = logging.getLogger("profile")

_KEYS = ("summary", "risk", "ladder", "health", "nextStep")
_MAX_LEN = 1200

_SYSTEM = (
    "You write a short financial-education explanation for a retail user in Indonesia. "
    "You receive facts computed by a rule engine. Rules: use ONLY numbers present in the facts "
    "(you may format them, e.g. 1,5 or 1.5 or Rp 37.500.000) and invent none; never recommend, "
    "name or rank any product, stock, fund, coin or ticker; never use the words buy, sell, beli "
    "or jual; this is education, not investment advice. Cover: summary; risk profile (say it is "
    "provisional when riskProfileIsProvisional is true); what the Financial Ladder level means; "
    "financial health (strongest and weakest dimension, category); ONE concrete next step based "
    "on the ladder level (level 1: emergency fund gap emergencyFundGapIDR; level 2: high-interest "
    "debt; level 3: start basic diversified investing; level 4: review allocation; level 5: "
    "estate and tax planning). Tone: warm, plain, no jargon, no emoji. Address the user by "
    "firstName if given. Reply with a single JSON object with exactly these string keys: "
    + ", ".join(_KEYS)
    + ". Do not include a disclaimer."
)


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
        {"role": "system", "content": _SYSTEM},
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
