import asyncio
from time import monotonic

import httpx


class OAuthTokenStore:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        token_uri: str,
    ):
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._token_uri = token_uri
        self._access_token: str | None = None
        self._expires_at = 0.0
        self._lock = asyncio.Lock()

    async def token(self, http: httpx.AsyncClient) -> str:
        async with self._lock:
            if self._access_token and monotonic() < self._expires_at - 60:
                return self._access_token

            response = await http.post(
                self._token_uri,
                data={
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "refresh_token": self._refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            response.raise_for_status()
            body = response.json()
            self._access_token = body["access_token"]
            self._expires_at = monotonic() + body.get("expires_in", 3600)
            return self._access_token

    async def apply(
        self,
        client: httpx.AsyncClient,
        token_http: httpx.AsyncClient,
    ) -> None:
        client.headers["Authorization"] = f"Bearer {await self.token(token_http)}"