import json
import logging
import re

from src.core.config import settings
from src.domain.user_profiling import facts as facts_mod
from src.domain.user_profiling import guardrail, template
from src.domain.user_profiling.schemas import ExplainRequest, ExplainResponse
from src.infrastructure.observability import tracing
from src.infrastructure.openrouter import LLMError, OpenRouterClient
from src.infrastructure.prompt.loader import load

logger = logging.getLogger("profile")

# the wording is shown as plain text, so markup or script links mean the model was steered
_MARKUP = re.compile(r"<\s*/?\s*[a-zA-Z!]|!\[|javascript:", re.IGNORECASE)
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
        if not isinstance(v, str) or not v.strip() or len(v) > _MAX_LEN or _MARKUP.search(v):
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
        content = await llm.complete(messages, purpose="explain_profile_wording")
    except LLMError:
        return None
    with tracing.span("parse_output", tracing.CHAIN, input=content) as pr:
        parts = _parse(content)
        pr.metadata(valid_json=parts is not None, fields=list(parts) if parts else None)
    if parts is None:
        logger.warning("llm output rejected: invalid json")
        return None
    allowed = facts_mod.numbers(f)
    with tracing.span("guardrail_check", tracing.GUARDRAIL, input=parts) as gc:
        rejected: tuple[str, str] | None = None
        for key, text in parts.items():
            bad = guardrail.violations(text, allowed)
            if bad:
                rejected = (key, bad[0])
                break
        gc.metadata(
            passed=rejected is None,
            rejected_field=rejected[0] if rejected else None,
            reason=rejected[1] if rejected else None,
            numbers_allowed=len(allowed),
        )
        gc.output("pass" if rejected is None else f"{rejected[0]}: {rejected[1]}")
    if rejected:
        logger.warning("llm output rejected: %s in %s", rejected[1], rejected[0])
        return None
    return parts


async def explain(req: ExplainRequest, llm: OpenRouterClient | None) -> ExplainResponse:
    with tracing.span(
        "explain_profile",
        tracing.AGENT,
        input=req.model_dump(mode="json"),
        track_totals=True,
        attributes={"tag.tags": ["profile_explanation", req.locale]},
    ) as root:
        with tracing.span("build_facts", tracing.CHAIN) as bf:
            f = facts_mod.build(req)
            bf.input(req.model_dump(mode="json"))
            # scores and levels only: no amounts and no name
            bf.metadata(
                locale=req.locale, level=f.level, profile=f.profile, category=f.category,
                total_score=f.total, provisional=f.provisional,
            )
            bf.output(facts_mod.prompt_facts(f))
        parts = await _generate(f, llm) if llm else None
        if parts is None:
            with tracing.span("template_fallback", tracing.CHAIN) as tf:
                texts = template.render(f)
                reason = "llm_not_configured" if llm is None else "llm_failed_or_rejected"
                tf.metadata(reason=reason)
                tf.output(texts)
        else:
            texts = parts
        resp = ExplainResponse(
            locale=req.locale,
            source="llm" if parts else "template",
            model=settings.openrouter_model if parts else None,
            disclaimer=template.DISCLAIMER[req.locale],
            **texts,
        )
        root.metadata(source=resp.source, locale=req.locale, model=resp.model)
        root.output(texts)
        root.ok()
    return resp
