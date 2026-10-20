from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# a chat with history and context is the largest legitimate request, and is far smaller
MAX_BODY_BYTES = 256 * 1024


class _TooLarge(Exception):
    pass


class BodyLimitMiddleware:
    """Refuses a body over the limit before it is parsed, announced or streamed."""

    def __init__(self, app: ASGIApp, limit: int = MAX_BODY_BYTES) -> None:
        self.app, self.limit = app, limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length", "")
        if declared.isdigit() and int(declared) > self.limit:
            await self._refuse(scope, receive, send)
            return

        seen = 0

        async def counted() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.limit:
                    raise _TooLarge
            return message

        started = False

        async def tracked(message: Message) -> None:
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, counted, tracked)
        except _TooLarge:
            if started:
                raise
            await self._refuse(scope, receive, send)

    @staticmethod
    async def _refuse(scope: Scope, receive: Receive, send: Send) -> None:
        from src.core.error_handler import error_response
        from src.core.errors import Code

        await error_response(Code.BAD_REQUEST, "Request body too large")(scope, receive, send)
