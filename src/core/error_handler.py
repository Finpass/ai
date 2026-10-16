import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.envelope import ErrorBody, Failure
from src.core.errors import STATUS_BY_CODE, AppError, Code, Detail
from src.core.request_context import request_id_var

logger = logging.getLogger("app")

_CODE_BY_STATUS = {status: code for code, status in STATUS_BY_CODE.items()}
_CODE_BY_STATUS[422] = Code.VALIDATION_FAILED


def error_response(code: Code, message: str, details: list[Detail] | None = None) -> JSONResponse:
    body = Failure(
        error=ErrorBody(code=code, message=message, details=details or []),
        requestId=request_id_var.get(),
    )
    return JSONResponse(status_code=STATUS_BY_CODE[code], content=body.model_dump(mode="json"))


def _validation_details(errors: list[Any]) -> list[Detail]:
    return [
        Detail(
            field=".".join(str(p) for p in e["loc"] if p != "body") or None,
            message=e["type"],
        )
        for e in errors
    ]


def add_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        if exc.status >= 500:
            logger.error("request failed", extra={"code": exc.code, "err": str(exc)})
        return error_response(exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            Code.VALIDATION_FAILED, "Validation failed", _validation_details(list(exc.errors()))
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _CODE_BY_STATUS.get(exc.status_code, Code.BAD_REQUEST)
        return error_response(code, str(exc.detail))
