import logging
import secrets
from contextvars import ContextVar

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "x-request-id"
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
logger = logging.getLogger("app")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        inbound = Headers(scope=scope).get(REQUEST_ID_HEADER, "")
        rid = inbound if 0 < len(inbound) <= 64 else secrets.token_hex(8)
        token = request_id_var.set(rid)

        started = False

        async def send_with_id(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = rid
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            if started:
                raise
            from src.core.error_handler import error_response
            from src.core.errors import Code

            logger.exception("unhandled exception")
            await error_response(Code.INTERNAL, "Internal server error")(
                scope, receive, send_with_id
            )
        finally:
            request_id_var.reset(token)
