"""The traces Phoenix receives: span names, kinds, nesting, tokens, cost, and what is withheld in production."""

import asyncio
import json

import httpx2 as httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.core.config import settings
from src.domain.advisor_chat.schemas import ChatRequest, UserContext
from src.domain.user_profiling.schemas import ExplainRequest
from src.features.advisor_chat import service
from src.features.user_profiling import service as profile_service
from src.infrastructure.advisor_chat import embedder, retrieval
from src.infrastructure.observability import tracing
from src.infrastructure.openrouter import OpenRouterClient
from tests.test_profile import completion, llm_json, payload
from tests.test_rag import FakeLLM, _passage

PRICED = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cost": 0.0004}


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing.install_tracer(provider.get_tracer("test"))
    monkeypatch.setattr(settings, "tracing_capture_content", True)
    yield exporter
    tracing.install_tracer(None)


def by_name(exporter):
    out = {}
    for s in exporter.get_finished_spans():
        out.setdefault(s.name, []).append(s)
    return out


def kind(s):
    return s.attributes["openinference.span.kind"]


def meta(s):
    return json.loads(s.attributes["metadata"])


def priced_client(reply="Bank wajib menjaga risiko [1].", seen=None):
    def handler(request):
        if seen is not None:
            seen.append(json.loads(request.content))
        body = {
            "id": "gen-1",
            "model": "anthropic/claude-3.5-haiku",
            "provider": "Anthropic",
            "choices": [{"message": {"content": reply}, "finish_reason": "stop"}],
            "usage": PRICED,
        }
        return httpx.Response(200, json=body)

    return OpenRouterClient(httpx.MockTransport(handler))


# ---------------------------------------------------------------------------------------- the LLM call


def test_llm_span_carries_tokens_cost_model_and_messages(spans):
    seen = []
    client = priced_client("halo", seen)
    out = asyncio.run(client.complete([{"role": "user", "content": "hai"}], purpose="uji"))
    assert out == "halo"
    assert seen[0]["usage"] == {"include": True}  # asks OpenRouter to report the cost

    (s,) = spans.get_finished_spans()
    a = s.attributes
    assert s.name == "uji" and kind(s) == "LLM"
    assert (
        a["llm.provider"] == "openrouter"
        and a["llm.response.model_name"] == "anthropic/claude-3.5-haiku"
    )
    assert (
        a["llm.token_count.prompt"],
        a["llm.token_count.completion"],
        a["llm.token_count.total"],
    ) == (100, 20, 120)
    assert a["llm.cost.total"] == pytest.approx(0.0004)
    assert a["llm.cost.prompt"] + a["llm.cost.completion"] == pytest.approx(0.0004)
    assert a["llm.input_messages.0.message.content"] == "hai"
    assert a["llm.output_messages.0.message.content"] == "halo"
    assert meta(s)["upstream_provider"] == "Anthropic" and meta(s)["finish_reason"] == "stop"


def test_failed_llm_call_is_an_error_span(spans):
    client = OpenRouterClient(httpx.MockTransport(lambda r: httpx.Response(500, json={})))
    with pytest.raises(Exception):  # noqa: B017  LLMError
        asyncio.run(client.complete([{"role": "user", "content": "hai"}]))
    (s,) = spans.get_finished_spans()
    assert s.status.status_code.name == "ERROR"


def test_content_is_withheld_when_capture_is_off(spans, monkeypatch):
    monkeypatch.setattr(settings, "tracing_capture_content", False)
    asyncio.run(
        priced_client("rahasia").complete([{"role": "user", "content": "gaji saya 20 juta"}])
    )
    (s,) = spans.get_finished_spans()
    assert not [k for k in s.attributes if "content" in k or k in ("input.value", "output.value")]
    assert s.attributes["llm.token_count.total"] == 120 and s.attributes["llm.cost.total"] > 0


def test_production_defaults_to_hidden_content(monkeypatch):
    monkeypatch.setattr(settings, "tracing_capture_content", None)
    monkeypatch.setattr(settings, "app_env", "production")
    assert tracing.capture_content() is False
    monkeypatch.setattr(settings, "app_env", "development")
    assert tracing.capture_content() is True


# ---------------------------------------------------------------------------------------- a whole chat turn


def test_chat_turn_is_one_tree_with_summed_cost(spans, monkeypatch):
    monkeypatch.setattr(service, "_search", lambda req: ([_passage(1)], "regulasi", []))
    req = ChatRequest(
        message="Apa kewajiban bank soal teknologi informasi?", userContext=UserContext()
    )
    resp = asyncio.run(service.chat(req, priced_client()))
    assert resp.source == "llm"

    n = by_name(spans)
    (root,) = n["advisor_chat"]
    assert kind(root) == "AGENT" and root.parent is None
    assert (
        root.attributes["input.value"].startswith("Apa kewajiban")
        and "[1]" in root.attributes["output.value"]
    )
    assert root.attributes["session.id"]  # hashed run id, for the Sessions view

    parent_of = {s.name: s.parent.span_id for ss in n.values() for s in ss if s.parent}
    for step in ("retrieve_context", "add_news_context", "generate_answer"):
        assert parent_of[step] == root.context.span_id, step
    (gen,) = n["generate_answer"]
    assert parent_of["answer_ready"] == gen.context.span_id  # the attempt that passed ends the turn
    (llm,) = n["chat_completion"]
    (guard,) = n["guardrail_check"]
    assert kind(llm) == "LLM" and llm.parent.span_id == gen.context.span_id
    assert (
        kind(guard) == "GUARDRAIL"
        and meta(guard)["passed"] is True
        and meta(guard)["citation_markers"] == [1]
    )

    # the answer call plus the progress-title calls all belong to the trace, and the root adds them up
    llm_spans = [s for s in spans.get_finished_spans() if kind(s) == "LLM"]
    assert len(llm_spans) >= 2 and any(s.name.startswith("turn_title:") for s in llm_spans)
    total = meta(root)
    assert total["llm_calls"] == len(llm_spans)
    assert total["cost_usd"] == pytest.approx(0.0004 * len(llm_spans))
    assert total["tokens_total"] == 120 * len(llm_spans)
    assert total["source"] == "llm" and total["grounded"] is True and total["intent"] == "regulasi"


def test_refused_answer_shows_both_attempts_and_the_fallback(spans, monkeypatch):
    monkeypatch.setattr(service, "_search", lambda req: ([_passage(1)], "regulasi", []))
    bad = "Lihat [9]."  # a marker that points nowhere
    resp = asyncio.run(
        service.chat(
            ChatRequest(message="Apa itu PADK?", userContext=UserContext()), FakeLLM([bad, bad])
        )
    )
    assert resp.source == "fallback"
    n = by_name(spans)
    attempts = sorted(n["generate_answer"], key=lambda s: meta(s)["attempt"])
    assert [meta(s)["attempt"] for s in attempts] == [1, 2] and meta(attempts[1])["retry_reason"]
    assert [meta(g)["passed"] for g in n["guardrail_check"]] == [False, False]
    assert "source_excerpts_fallback" in n and "answer_ready" not in n


def test_error_in_the_pipeline_marks_the_root_failed(spans, monkeypatch):
    def boom(req):
        raise RuntimeError("chroma down")

    monkeypatch.setattr(service, "_search", boom)
    with pytest.raises(RuntimeError):
        asyncio.run(
            service.chat(ChatRequest(message="halo dunia", userContext=UserContext()), None)
        )
    root = by_name(spans)["advisor_chat"][0]
    assert root.status.status_code.name == "ERROR"
    assert by_name(spans)["retrieve_context"][0].status.status_code.name == "ERROR"


# ---------------------------------------------------------------------------------------- retrieval stages


def test_retrieval_stages_are_spans_with_documents_and_scores(spans, monkeypatch):
    passages = [_passage(1), _passage(2)]
    passages[1].content += " Aturan kedua."  # identical text would be merged as a duplicate
    monkeypatch.setattr(retrieval, "known_tickers", lambda: {"BBCA"})
    monkeypatch.setattr(retrieval, "_candidates", lambda *a, **k: list(passages))
    monkeypatch.setattr(embedder, "rerank", lambda q, texts: [0.91, 0.40])

    with tracing.span("parent"):
        got, intent, _ = retrieval.search(
            "Apa aturan OJK soal bank?", "Apa aturan OJK soal bank?", "id", []
        )
    assert intent == "regulasi" and len(got) == 2

    n = by_name(spans)
    assert (
        kind(n["plan_retrieval"][0]) == "CHAIN"
        and meta(n["plan_retrieval"][0])["intent"] == "regulasi"
    )
    vs = n["vector_search"][0]
    assert kind(vs) == "RETRIEVER" and vs.attributes["retrieval.documents.0.document.id"] == "1"
    assert "retrieval.documents.1.document.content" in vs.attributes
    rr = n["rerank_passages"][0]
    assert kind(rr) == "RERANKER" and rr.attributes["reranker.model_name"] == settings.rerank_model
    assert rr.attributes["reranker.input_documents.1.document.id"] == "2"
    assert rr.attributes["reranker.output_documents.0.document.score"] == pytest.approx(0.91)
    sel = meta(n["select_passages"][0])
    assert sel["kept"] == 2 and sel["above_cutoff"] == 2 and sel["top_score"] == pytest.approx(0.91)
    assert meta(n["gather_candidates"][0])["candidates"] == 2


def test_embedding_span_reports_model_tokens_and_cost(spans, monkeypatch):
    class Resp:
        status_code = 200

        def json(self):
            vec = [1.0] + [0.0] * (embedder.EMBED_DIMENSIONS - 1)
            return {
                "data": [{"index": 0, "embedding": vec}],
                "usage": {"prompt_tokens": 7, "total_tokens": 7, "cost": 1e-7},
            }

    class Http:
        def __init__(self, **kw): ...
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **k):
            return Resp()

    monkeypatch.setattr(embedder.httpx, "Client", Http)
    monkeypatch.setattr(embedder, "_api_key", lambda: "k")
    with tracing.span("parent", track_totals=True) as parent:
        embedder.embed(["apa itu OJK"], fast=True)
    (e,) = by_name(spans)["embed_query"]
    assert kind(e) == "EMBEDDING" and e.attributes["embedding.model_name"] == embedder.EMBED_MODEL
    assert e.attributes["llm.token_count.total"] == 7 and e.attributes[
        "llm.cost.total"
    ] == pytest.approx(1e-7)
    assert e.attributes["embedding.embeddings.0.embedding.text"] == "apa itu OJK"
    assert parent  # handle returned


# ---------------------------------------------------------------------------------------- profile explanation


def test_profile_explanation_is_traced_with_its_guardrail(spans):
    reply = json.dumps(llm_json())
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={**completion(reply), "usage": PRICED, "model": "m"})

    client = OpenRouterClient(httpx.MockTransport(handler))
    resp = asyncio.run(profile_service.explain(ExplainRequest(**payload()), client))
    n = by_name(spans)
    root = n["explain_profile"][0]
    assert kind(root) == "AGENT" and meta(root)["source"] == resp.source
    assert kind(n["explain_profile_wording"][0]) == "LLM"
    assert kind(n["guardrail_check"][0]) == "GUARDRAIL"
    assert meta(n["build_facts"][0])["level"] == 1
    assert "Dinda" not in json.dumps(meta(n["build_facts"][0]))  # no name in metadata
    assert meta(root)["cost_usd"] == pytest.approx(0.0004)


def test_profile_without_llm_traces_the_template(spans):
    resp = asyncio.run(profile_service.explain(ExplainRequest(**payload()), None))
    assert resp.source == "template"
    n = by_name(spans)
    assert meta(n["template_fallback"][0])["reason"] == "llm_not_configured"
    assert "explain_profile_wording" not in n


# ---------------------------------------------------------------------------------------- switching it off


def test_everything_is_a_noop_when_tracing_is_off():
    tracing.install_tracer(None)
    with tracing.span("x", tracing.AGENT, input="q", track_totals=True) as t:
        t.metadata(a=1)
        t.output("o")
        t.usage(PRICED, kind=tracing.LLM)
        tracing.current().attr("k", "v")
    assert tracing.enabled() is False
    assert asyncio.run(priced_client("ok").complete([{"role": "user", "content": "hai"}])) == "ok"


def test_setup_needs_an_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "phoenix_collector_endpoint", "")
    monkeypatch.setattr(settings, "tracing_enabled", None)
    assert settings.tracing_active is False and tracing.setup_tracing() is False
    monkeypatch.setattr(settings, "phoenix_collector_endpoint", "http://localhost:6006")
    assert settings.tracing_active is True
    monkeypatch.setattr(settings, "tracing_enabled", False)
    assert settings.tracing_active is False  # an explicit off wins


@pytest.mark.parametrize(
    "given,url,protocol",
    [
        ("http://localhost:6006", "http://localhost:6006/v1/traces", "http/protobuf"),
        ("http://localhost:6006/", "http://localhost:6006/v1/traces", "http/protobuf"),
        ("http://localhost:6006/v1/traces", "http://localhost:6006/v1/traces", "http/protobuf"),
        (
            "https://app.phoenix.arize.com/s/finpass",
            "https://app.phoenix.arize.com/s/finpass/v1/traces",
            "http/protobuf",
        ),
        ("http://phoenix:4317", "http://phoenix:4317", "grpc"),
    ],
)
def test_collector_url_is_normalized(given, url, protocol):
    assert tracing.collector_url(given) == (url, protocol)
