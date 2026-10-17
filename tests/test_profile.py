import copy
import json

import httpx2 as httpx
import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.core.config import DEV_INTERNAL_KEY, settings
from src.domain.user_profiling import facts as facts_mod
from src.domain.user_profiling import guardrail
from src.domain.user_profiling.schemas import ExplainRequest
from src.features.user_profiling.router import get_llm
from src.infrastructure.openrouter import OpenRouterClient

KEY = {"X-Internal-Api-Key": DEV_INTERNAL_KEY}
URL = "/ml/profile/explain"

BASE = {
    "locale": "id",
    "persona": {"firstName": "Dinda"},
    "result": {
        "capacityScore": 55.5,
        "willingnessScore": 62,
        "finalProfile": "Moderate",
        "profileLabel": "provisional",
        "riskScoreEquivalent": 5.5,
        "ladderLevel": 1,
        "ladderProgress": 20,
        "emergencyFundMonths": 1.5,
        "emergencyTargetMonths": {"min": 6, "max": 6},
        "flags": [{"code": "provisional", "messageEn": "x", "messageId": "y"}],
        "financialHealth": {
            "financialSecurity": 30,
            "financialSafety": 45.5,
            "financialGrowth": 60,
            "financialWisdom": 70,
            "financialFreedom": 20,
            "totalScore": 45.1,
            "category": "Coping",
        },
        "metrics": {
            "monthlyIncome": 15_000_000,
            "monthlyExpenses": 10_000_000,
            "monthlyInstallment": 1_000_000,
            "monthlySurplus": 4_000_000,
            "savingsRate": 0.25,
            "dti": 0.1,
            "investableRatio": 0.2,
        },
    },
    "answers": {"q4": 15_000_000},
}
SECTIONS = ("summary", "risk", "ladder", "health", "nextStep")


def payload(locale="id", level=1, profile="Moderate", **over):
    p = copy.deepcopy(BASE)
    p["locale"] = locale
    p["result"]["ladderLevel"] = level
    p["result"]["finalProfile"] = profile
    p.update(over)
    return p


def llm_json(**over):
    d = {
        "summary": "Dinda, profil Anda Moderat dan Anda di level 1.",
        "risk": "Profil risiko Anda Moderat, masih sementara.",
        "ladder": "Level 1 berarti membangun dana darurat; Anda punya 1,5 bulan.",
        "health": "Skor total 45,1 dari 100; terkuat Wisdom 70, terlemah Freedom 20.",
        "nextStep": "Kekurangan dana darurat Rp 45.000.000 dari target Rp 60.000.000.",
    }
    d.update(over)
    return d


def completion(content):
    return {"choices": [{"message": {"content": content}}]}


@pytest.fixture
def make_client():
    def _make(handler=None):
        if handler is None:
            app.dependency_overrides[get_llm] = lambda: None
        else:
            client = OpenRouterClient(httpx.MockTransport(handler))
            app.dependency_overrides[get_llm] = lambda: client
        return TestClient(app)

    yield _make
    app.dependency_overrides.clear()


def ok(c, **kw):
    r = c.post(URL, json=payload(**kw), headers=KEY)
    assert r.status_code == 200, r.text
    return r.json()["data"]


@pytest.mark.parametrize("locale", ["id", "en"])
@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("profile", ["Conservative", "Moderate", "Aggressive"])
def test_template_complete_and_passes_guardrail(make_client, locale, level, profile):
    d = ok(make_client(), locale=locale, level=level, profile=profile)
    assert d["source"] == "template" and d["model"] is None and d["locale"] == locale
    assert "provisional" in d["risk"] or "sementara" in d["risk"]
    req = ExplainRequest.model_validate(payload(locale, level))
    allowed = facts_mod.numbers(facts_mod.build(req))
    for k in SECTIONS:
        assert d[k]
        assert guardrail.violations(d[k], allowed) == [], (k, d[k])
    assert d["disclaimer"] == (
        "Ini panduan edukasi, bukan nasihat investasi."
        if locale == "id"
        else "This is financial guidance and education, not investment advice."
    )


def test_template_gap_and_formats(make_client):
    d = ok(make_client())
    assert "Rp 45.000.000" in d["nextStep"] and "1,5" in d["nextStep"]
    assert d["summary"].startswith("Dinda,")
    e = ok(make_client(), locale="en")
    assert "Rp 45,000,000" in e["nextStep"] and "1.5" in e["nextStep"]
    assert "Wisdom" in e["health"] and "Freedom" in e["health"]


def test_savings_falls_back_to_months_times_expenses(make_client):
    d = ok(make_client(), answers={})
    assert "Rp 45.000.000" in d["nextStep"]


def test_firstname_sanitized(make_client):
    d = ok(make_client(), persona={"firstName": "  Dinda 99 Putri"})
    assert d["summary"].startswith("Dinda,")
    d = ok(make_client(), persona={})
    assert not d["summary"].startswith(",")


# guardrail
ALLOWED = facts_mod.numbers(facts_mod.build(ExplainRequest.model_validate(BASE)))


@pytest.mark.parametrize(
    "text",
    [
        "Anda punya 1,5 bulan",
        "You hold 1.5 months",
        "Rp 60.000.000 tersisa",
        "Rp 45,000,000 short",
        "Skor 45,1 dari 100",
        "rasio 25% dan 10%",
    ],
)
def test_guardrail_accepts(text):
    assert guardrail.violations(text, ALLOWED) == []


@pytest.mark.parametrize(
    "text",
    [
        "Anda punya 2 bulan",
        "Rp 60.000.001",
        "skor 99",
        "Sebaiknya beli saham",
        "You should buy more",
        "Time to SELL now",
        "Jual aset Anda",
        "Pertimbangkan BBCA",
        "Bitcoin menarik",
    ],
)
def test_guardrail_rejects(text):
    assert guardrail.violations(text, ALLOWED)


# LLM path and fallbacks
def test_llm_success(make_client):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion(json.dumps(llm_json())))

    app.dependency_overrides.clear()
    d = ok(make_client(handler))
    assert d["source"] == "llm" and d["model"] == settings.openrouter_model
    assert d["summary"].startswith("Dinda, profil")
    assert d["disclaimer"].startswith("Ini panduan")
    body = json.dumps(seen["body"])
    assert "Dinda" in body and "answers" not in body and "15000000" not in body.replace(".0", "")
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_llm_fenced_json_ok(make_client):
    def h(r):
        fenced = "```json\n" + json.dumps(llm_json()) + "\n```"
        return httpx.Response(200, json=completion(fenced))

    assert ok(make_client(h))["source"] == "llm"


def _boom(request):
    raise httpx.ConnectTimeout("t", request=request)


@pytest.mark.parametrize(
    "handler",
    [
        _boom,
        lambda r: httpx.Response(500, json={"error": "x"}),
        lambda r: httpx.Response(200, text="not json"),
        lambda r: httpx.Response(200, json={"choices": []}),
        lambda r: httpx.Response(200, json=completion("plain prose")),
        lambda r: httpx.Response(200, json=completion(json.dumps({"summary": "x"}))),
        lambda r: httpx.Response(200, json=completion(json.dumps(llm_json(risk="")))),
        lambda r: httpx.Response(200, json=completion(json.dumps(llm_json(nextStep="Beli saham")))),
        lambda r: httpx.Response(200, json=completion(json.dumps(llm_json(health="Skor 77")))),
    ],
    ids=[
        "timeout",
        "http500",
        "badbody",
        "nochoices",
        "prose",
        "missingkeys",
        "empty",
        "advice",
        "badnumber",
    ],
)
def test_fallback_to_template(make_client, handler):
    d = ok(make_client(handler))
    assert d["source"] == "template" and d["model"] is None
    assert all(d[k] for k in SECTIONS)


def test_no_key_means_template(make_client, monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    app.dependency_overrides.clear()
    assert ok(TestClient(app))["source"] == "template"


def test_key_never_logged(make_client, monkeypatch, caplog):
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-secret-123")
    make = make_client(lambda r: httpx.Response(500, text="sk-secret-123 Dinda"))
    with caplog.at_level("DEBUG"):
        ok(make)
    assert "sk-secret-123" not in caplog.text and "Dinda" not in caplog.text


# HTTP contract
def test_auth_required(make_client):
    c = make_client()
    assert c.post(URL, json=payload()).status_code == 401
    assert c.post(URL, json=payload(), headers={"X-Internal-Api-Key": "bad"}).status_code == 401


def test_validation_errors(make_client):
    c = make_client()
    for bad in (
        {**payload(), "locale": "fr"},
        {**payload(), "extra": 1},
        {k: v for k, v in payload().items() if k != "result"},
        payload(level=9),
    ):
        r = c.post(URL, json=bad, headers=KEY)
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "VALIDATION_FAILED"
    p = payload()
    p["result"]["financialHealth"]["category"] = "Nope"
    assert c.post(URL, json=p, headers=KEY).status_code == 422
