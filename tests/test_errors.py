from src.core.config import settings
from src.infrastructure.observability import errors


def test_scrub_removes_secrets_and_bodies():
    event = {
        "request": {
            "headers": {"X-Internal-Api-Key": "k", "Authorization": "b", "Accept": "x"},
            "cookies": {"a": "b"},
            "data": {"message": "gaji saya 20 juta"},
            "query_string": "since=0",
        }
    }
    out = errors._scrub(event, {})["request"]
    assert out["headers"] == {"Accept": "x"}
    assert not {"cookies", "data", "query_string"} & set(out)


def test_off_without_dsn(monkeypatch):
    monkeypatch.setattr(settings, "sentry_dsn", "")
    assert errors.setup_error_reporting() is False


def test_bad_dsn_does_not_raise(monkeypatch):
    monkeypatch.setattr(settings, "sentry_dsn", "not-a-dsn")
    assert errors.setup_error_reporting() is False
