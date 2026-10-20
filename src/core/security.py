from hmac import compare_digest
from typing import Annotated

from fastapi import Depends, Header

from src.core.config import settings
from src.core.errors import unauthorized


async def _check_key(x_internal_api_key: Annotated[str | None, Header()] = None) -> None:
    if not x_internal_api_key:
        raise unauthorized("Invalid internal API key")
    accepted = [k for k in (settings.internal_api_key, settings.internal_api_key_previous) if k]
    given = x_internal_api_key.encode()
    # compare against every accepted key, so timing does not tell which one matched
    if not any([compare_digest(given, k.encode()) for k in accepted]):
        raise unauthorized("Invalid internal API key")


require_internal_key = Depends(_check_key)
