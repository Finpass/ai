import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.core.config import DEV_INTERNAL_KEY

KEY = {"X-Internal-Api-Key": DEV_INTERNAL_KEY}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def err(resp) -> dict:
    body = resp.json()
    assert body["success"] is False
    assert resp.headers["x-request-id"] == body["requestId"]
    return body["error"]


def test_health_envelope(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"success": True, "message": "ok", "data": None}


def test_missing_key_is_unauthorized(client):
    r = client.post("/ml/profile/explain", json={})
    assert r.status_code == 401
    assert err(r)["code"] == "UNAUTHORIZED"


def test_validation_details(client):
    r = client.post("/ml/profile/explain", json={}, headers=KEY)
    e = err(r)
    assert r.status_code == 422
    assert e["code"] == "VALIDATION_FAILED"
    assert e["details"]


def test_unknown_route_and_method(client):
    assert err(client.get("/nope"))["code"] == "NOT_FOUND"
    r = client.delete("/health")
    assert r.status_code == 405
    assert err(r)["code"] == "METHOD_NOT_ALLOWED"


def test_unhandled_exception_hides_detail_and_keeps_request_id():
    @app.get("/_boom")
    async def boom():
        raise RuntimeError("secret detail")

    r = TestClient(app).get("/_boom", headers={"X-Request-Id": "abc"})
    e = err(r)
    assert r.status_code == 500
    assert e == {"code": "INTERNAL", "message": "Internal server error", "details": []}
    assert r.headers["x-request-id"] == "abc"
