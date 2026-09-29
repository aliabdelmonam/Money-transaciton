import hmac
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    request: Request, provided: Optional[str] = Security(api_key_header)
) -> None:
    """Allow the request only if ``X-API-Key`` matches ``API_KEY``."""
    expected: Optional[str] = request.app.state.api_key
    if not expected or not provided or not hmac.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="invalid or missing API key")
