"""Sentry error reporting. Off unless SENTRY_DSN is set; never raises.

Prompts, answers and the user's finances must not reach a third party, so the SDK is told not to
collect them (no personal data, no request bodies, no local variables) and `_scrub` removes the
secrets we send in headers.
Performance tracing stays off: the traces go to Phoenix (tracing.py).
"""

import logging
from typing import Any

from src.core.config import settings

logger = logging.getLogger("errors")

_SECRET_HEADERS = {"authorization", "cookie", "x-internal-api-key", "x-api-key"}


def _scrub(event: dict[str, Any], _hint: dict[str, Any]) -> dict[str, Any]:
    request = event.get("request") or {}
    headers = request.get("headers")
    if isinstance(headers, dict):
        request["headers"] = {k: v for k, v in headers.items() if k.lower() not in _SECRET_HEADERS}
    request.pop("cookies", None)
    request.pop("data", None)
    request.pop("query_string", None)  # run ids carry the user id
    return event


def setup_error_reporting() -> bool:
    if not settings.sentry_dsn:
        logger.info("error reporting off (set SENTRY_DSN to turn it on)")
        return False
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.app_env,
            release=settings.service_version,
            send_default_pii=False,
            include_local_variables=False,  # frames would carry prompts and user facts
            max_request_body_size="never",
            before_send=_scrub,
        )
    except Exception:  # error reporting must never stop the service from starting
        logger.exception("sentry setup failed; continuing without it")
        return False
    logger.info("error reporting on")
    return True
