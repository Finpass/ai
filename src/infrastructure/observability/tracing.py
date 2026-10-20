"""Arize Phoenix tracing: OpenTelemetry spans that follow the OpenInference conventions.

One chat turn becomes one trace shaped like this (kind in brackets):

    advisor_chat                       [AGENT]      input = question, output = answer, totals for tokens and cost
       ├─ retrieve_context                [CHAIN]
       │  ├─ plan_retrieval               [CHAIN]      intent, tickers, source types, years, expanded query
       │  ├─ gather_candidates            [CHAIN]
       │  │  └─ vector_search             [RETRIEVER]  Chroma filters, returned documents
       │  │     └─ embed_query            [EMBEDDING]  model, tokens, cost
       │  ├─ rerank_passages              [RERANKER]   model, input and output documents with scores
       │  └─ select_passages              [CHAIN]      cutoff, year balance, top n
       ├─ add_news_context                [CHAIN]      only when fewer than 3 documents were found
       ├─ generate_answer (attempt 1, 2)  [CHAIN]
       │  ├─ chat_completion              [LLM]        messages, tokens, cost, finish reason
       │  └─ guardrail_check              [GUARDRAIL]  violations, citation markers
       ├─ source_excerpts_fallback        [CHAIN]      only when the answer was refused twice
       └─ turn_title:<step> ...           [LLM]        the small calls that write progress titles

The profile explanation is traced the same way under `explain_profile`. There are no HTTP server spans: they
had no input or output, and the polling endpoint for turn titles flooded the project.

Tracing never changes behavior: with it off, or if anything in here fails, every helper is a no-op. Content
(questions, prompts, answers, user facts) is exported only when `capture_content` is on. It defaults to on in
development and off in production, because prompts carry a person's financial data. Metadata (model, tokens, cost,
document ids and titles, scores, timings) is always exported.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.core.config import settings

logger = logging.getLogger("tracing")

SERVICE_NAME = "finpass-ai"
_TRACER_NAME = "finpass.ai"
_tracer: trace.Tracer | None = None

# OpenInference span kinds
AGENT, CHAIN, LLM, RETRIEVER, RERANKER, EMBEDDING, GUARDRAIL = (
    "AGENT",
    "CHAIN",
    "LLM",
    "RETRIEVER",
    "RERANKER",
    "EMBEDDING",
    "GUARDRAIL",
)

_DOC_CONTENT_LIMIT = 700
_TEXT_LIMIT = 20_000


# ------------------------------------------------------------------------------------------------ setup


def collector_url(endpoint: str) -> tuple[str, str]:
    """Turns what people put in PHOENIX_COLLECTOR_ENDPOINT into the exact OTLP URL and the protocol to use.

    `http://localhost:6006` and a Phoenix Cloud space URL get `/v1/traces` added; a full traces URL is kept; a
    gRPC port (4317) is kept and exported over gRPC.
    """
    url = endpoint.strip().rstrip("/")
    if url.endswith("/v1/traces"):
        return url, "http/protobuf"
    if url.rsplit(":", 1)[-1].isdigit() and url.rsplit(":", 1)[-1] == "4317":
        return url, "grpc"
    return url + "/v1/traces", "http/protobuf"


def setup_tracing() -> bool:
    """Connects the service to Phoenix. Returns True when tracing is on. Never raises."""
    global _tracer
    if not settings.tracing_active:
        logger.info("tracing off (set PHOENIX_COLLECTOR_ENDPOINT to turn it on)")
        return False
    try:
        from phoenix.otel import register

        os.environ.setdefault(
            "OTEL_RESOURCE_ATTRIBUTES",
            f"service.name={SERVICE_NAME},service.version={settings.service_version},"
            f"deployment.environment={settings.app_env}",
        )
        endpoint, protocol = collector_url(settings.phoenix_collector_endpoint)
        provider = register(
            endpoint=endpoint,
            protocol=protocol,
            project_name=settings.phoenix_project_name,
            api_key=settings.phoenix_api_key or None,
            batch=True,  # export in the background so a slow collector never slows an answer
            auto_instrument=False,  # we trace by hand: LLM calls use httpx, not an SDK that could be patched
            verbose=False,
        )
        _tracer = provider.get_tracer(_TRACER_NAME)
        logger.info(
            "tracing on: project=%s content=%s",
            settings.phoenix_project_name,
            "captured" if capture_content() else "hidden",
        )
        return True
    except Exception:  # tracing must never stop the service from starting
        logger.exception("tracing setup failed; continuing without it")
        _tracer = None
        return False


def install_tracer(tracer: trace.Tracer | None) -> None:
    """Test hook: route spans to a tracer of the caller's choice (or None to switch tracing off)."""
    global _tracer
    _tracer = tracer


def enabled() -> bool:
    return _tracer is not None


def capture_content() -> bool:
    flag = settings.tracing_capture_content
    return flag if flag is not None else not settings.is_production


# ------------------------------------------------------------------------------------------------ run totals


@dataclass
class RunTotals:
    """What one agent run spent, summed over every LLM and embedding call made inside it."""

    llm_calls: int = 0
    embedding_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    priced_calls: int = 0  # calls for which the provider reported a cost


_totals: ContextVar[RunTotals | None] = ContextVar("tracing_totals", default=None)
_current: ContextVar[Trace | None] = ContextVar("tracing_current", default=None)


# ------------------------------------------------------------------------------------------------ span wrapper


class Trace:
    """Thin handle over one span. Every method is safe to call whether or not tracing is on."""

    def __init__(self, span: trace.Span | None) -> None:
        self._span = span
        self._meta: dict[str, Any] = {}
        self._tokens_prompt = 0
        self._tokens_total = 0
        self._cost = 0.0
        self._priced = False

    # generic
    def attr(self, key: str, value: Any) -> None:
        if self._span is None or value is None:
            return
        try:
            self._span.set_attribute(key, _scalar(value))
        except Exception:  # pragma: no cover - the SDK swallows bad values, this is belt and braces
            logger.debug("bad span attribute %s", key)

    def attrs(self, values: Mapping[str, Any]) -> None:
        for k, v in values.items():
            self.attr(k, v)

    def metadata(self, **values: Any) -> None:
        self._meta.update({k: v for k, v in values.items() if v is not None})

    def input(self, value: Any, *, mime: str | None = None) -> None:
        self._io("input", value, mime)

    def output(self, value: Any, *, mime: str | None = None) -> None:
        self._io("output", value, mime)

    def _io(self, side: str, value: Any, mime: str | None) -> None:
        if self._span is None or value is None:
            return
        text, auto = (
            (value, "text/plain") if isinstance(value, str) else (_json(value), "application/json")
        )
        if not capture_content():  # never leave the column empty: say what was withheld
            text = f"[content hidden: {len(text)} chars; TRACING_CAPTURE_CONTENT=true shows it]"
            auto = "text/plain"
        self.attr(f"{side}.value", _clip(text, _TEXT_LIMIT))
        self.attr(f"{side}.mime_type", mime or auto)

    def fail(self, exc: BaseException) -> None:
        if self._span is None:
            return
        self._span.record_exception(exc)
        self._span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"[:300]))

    def ok(self) -> None:
        if self._span is not None:
            self._span.set_status(Status(StatusCode.OK))

    # llm and embedding specifics
    def usage(self, usage: Mapping[str, Any] | None, *, kind: str) -> None:
        """Records the token counts and the cost the provider reported, and adds them to the run totals."""
        if not usage:
            return
        prompt = _int(usage.get("prompt_tokens"))
        completion = _int(usage.get("completion_tokens"))
        total = _int(usage.get("total_tokens")) or prompt + completion
        cost = usage.get("cost")
        self._tokens_prompt, self._tokens_total = prompt, total
        self.attr("llm.token_count.prompt", prompt)
        self.attr("llm.token_count.total", total)
        if kind == LLM:
            self.attr("llm.token_count.completion", completion)
        if isinstance(cost, int | float):
            self._cost, self._priced = float(cost), True
            self.attr("llm.cost.total", float(cost))
            if (
                kind == LLM and total
            ):  # split in proportion to tokens when the provider gives only a total
                self.attr("llm.cost.prompt", float(cost) * prompt / total)
                self.attr("llm.cost.completion", float(cost) * completion / total)
        run = _totals.get()
        if run is not None:
            run.prompt_tokens += prompt
            run.completion_tokens += completion
            run.cost_usd += self._cost
            run.priced_calls += 1 if self._priced else 0
            if kind == LLM:
                run.llm_calls += 1
            else:
                run.embedding_calls += 1

    def _finish(self) -> None:
        if self._span is not None and self._meta:
            self.attr("metadata", _json(self._meta))


_NULL = Trace(None)


def current() -> Trace:
    """The innermost open span, so deep code can add an attribute without being passed a handle."""
    return _current.get() or _NULL


@contextmanager
def span(
    name: str,
    kind: str = CHAIN,
    *,
    input: Any = None,
    attributes: Mapping[str, Any] | None = None,
    track_totals: bool = False,
) -> Iterator[Trace]:
    """Opens a span named `name` of OpenInference kind `kind`. An exception marks the span as failed and propagates."""
    if _tracer is None:
        yield _NULL
        return
    totals_token = _totals.set(RunTotals()) if track_totals else None
    with _tracer.start_as_current_span(
        name, record_exception=False, set_status_on_exception=False
    ) as sp:
        t = Trace(sp)
        t.attr("openinference.span.kind", kind)
        if attributes:
            t.attrs(attributes)
        t.input(input)
        cur_token = _current.set(t)
        try:
            yield t
        except BaseException as exc:
            t.fail(exc)
            raise
        finally:
            _current.reset(cur_token)
            if totals_token is not None:
                run = _totals.get()
                if run is not None:
                    t.metadata(
                        llm_calls=run.llm_calls,
                        embedding_calls=run.embedding_calls,
                        tokens_prompt=run.prompt_tokens,
                        tokens_completion=run.completion_tokens,
                        tokens_total=run.prompt_tokens + run.completion_tokens,
                        cost_usd=round(run.cost_usd, 8),
                        priced_calls=run.priced_calls,
                    )
                _totals.reset(totals_token)
            t._finish()


# ------------------------------------------------------------------------------------------------ typed helpers


@contextmanager
def llm_span(
    name: str,
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    purpose: str,
) -> Iterator[Trace]:
    """A span for one chat completion. Call `llm_result` on the handle when the answer is in."""
    with span(
        name,
        LLM,
        attributes={
            "llm.model_name": model,
            "llm.provider": "openrouter",
            "llm.system": "openrouter",
            "llm.invocation_parameters": _json(dict(params)),
        },
    ) as t:
        t.metadata(purpose=purpose)
        t.attr("llm.input_messages.count", len(messages))
        if capture_content():
            for i, m in enumerate(messages):
                t.attr(f"llm.input_messages.{i}.message.role", m.get("role"))
                t.attr(
                    f"llm.input_messages.{i}.message.content",
                    _clip(str(m.get("content", "")), _TEXT_LIMIT),
                )
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if last_user:
            t.input(str(last_user.get("content", "")))
        yield t


def llm_result(t: Trace, content: str, data: Mapping[str, Any]) -> None:
    """Fills an LLM span from an OpenRouter chat completion response."""
    choice = (data.get("choices") or [{}])[0]
    t.attr("llm.response.model_name", data.get("model"))
    t.metadata(
        upstream_provider=data.get("provider"),
        finish_reason=choice.get("finish_reason"),
        response_id=data.get("id"),
    )
    t.usage(data.get("usage"), kind=LLM)
    if capture_content():
        t.attr("llm.output_messages.0.message.role", "assistant")
        t.attr("llm.output_messages.0.message.content", _clip(content, _TEXT_LIMIT))
    t.output(content)
    t.ok()


@contextmanager
def embedding_span(texts: Sequence[str], *, model: str, dimensions: int) -> Iterator[Trace]:
    with span(
        "embed_query",
        EMBEDDING,
        attributes={
            "embedding.model_name": model,
            "embedding.invocation_parameters": _json({"dimensions": dimensions}),
            "llm.provider": "openrouter",
        },
    ) as t:
        t.metadata(texts=len(texts), dimensions=dimensions)
        if capture_content():
            for i, text in enumerate(texts):
                t.attr(f"embedding.embeddings.{i}.embedding.text", _clip(text, 2000))
        t.input(list(texts) if len(texts) > 1 else texts[0])
        yield t


def documents(t: Trace, prefix: str, docs: Sequence[Mapping[str, Any]]) -> None:
    """Writes documents under `retrieval.documents` or `reranker.input_documents` / `reranker.output_documents`."""
    for i, d in enumerate(docs):
        base = f"{prefix}.{i}.document"
        t.attr(f"{base}.id", str(d.get("id", "")))
        t.attr(f"{base}.score", d.get("score"))
        content = d.get("content")
        if (
            content is not None
        ):  # source documents are public text, so content is exported even when prompts are not
            t.attr(f"{base}.content", _clip(str(content), _DOC_CONTENT_LIMIT))
        meta = {
            k: v
            for k, v in d.items()
            if k not in ("id", "score", "content") and v not in (None, "")
        }
        if meta:
            t.attr(f"{base}.metadata", _json(meta))


def session_id(run_id: str | None) -> str | None:
    """A stable id for the Phoenix Sessions view. Hashed: the run id carries the user id."""
    return hashlib.sha256(run_id.encode()).hexdigest()[:16] if run_id else None


# ------------------------------------------------------------------------------------------------ small utilities


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _int(v: Any) -> int:
    return int(v) if isinstance(v, int | float) else 0


def _json(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, default=str)


def _scalar(v: Any) -> Any:
    if isinstance(v, str | bool | int | float):
        return v
    if isinstance(v, list | tuple) and all(isinstance(x, str | bool | int | float) for x in v):
        return list(v)
    return _json(v)
