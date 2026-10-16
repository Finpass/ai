from pydantic import BaseModel

from src.core.errors import Code, Detail


class Success[T](BaseModel):
    success: bool = True
    message: str = "ok"
    data: T


class ErrorBody(BaseModel):
    code: Code
    message: str
    details: list[Detail]


class Failure(BaseModel):
    success: bool = False
    error: ErrorBody
    requestId: str | None = None
