from enum import StrEnum

from pydantic import BaseModel


class Code(StrEnum):
    BAD_REQUEST = "BAD_REQUEST"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    CONFLICT = "CONFLICT"
    UNPROCESSABLE = "UNPROCESSABLE"
    TOO_MANY_REQUESTS = "TOO_MANY_REQUESTS"
    INTERNAL = "INTERNAL"
    UPSTREAM_FAILED = "UPSTREAM_FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    UPSTREAM_TIMEOUT = "UPSTREAM_TIMEOUT"


STATUS_BY_CODE: dict[Code, int] = {
    Code.BAD_REQUEST: 400,
    Code.VALIDATION_FAILED: 422,
    Code.UNAUTHORIZED: 401,
    Code.FORBIDDEN: 403,
    Code.NOT_FOUND: 404,
    Code.METHOD_NOT_ALLOWED: 405,
    Code.CONFLICT: 409,
    Code.UNPROCESSABLE: 422,
    Code.TOO_MANY_REQUESTS: 429,
    Code.INTERNAL: 500,
    Code.UPSTREAM_FAILED: 502,
    Code.UNAVAILABLE: 503,
    Code.UPSTREAM_TIMEOUT: 504,
}


class Detail(BaseModel):
    field: str | None = None
    message: str


class AppError(Exception):
    def __init__(self, code: Code, message: str, details: list[Detail] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or []

    @property
    def status(self) -> int:
        return STATUS_BY_CODE[self.code]


def not_found(message: str = "Resource not found") -> AppError:
    return AppError(Code.NOT_FOUND, message)


def unauthorized(message: str = "Unauthorized") -> AppError:
    return AppError(Code.UNAUTHORIZED, message)


def unprocessable(message: str) -> AppError:
    return AppError(Code.UNPROCESSABLE, message)


def upstream_failed(message: str) -> AppError:
    return AppError(Code.UPSTREAM_FAILED, message)
