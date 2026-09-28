from typing import Callable, Optional

import httpx

from channels.models.health import ChannelHealth


async def http_probe(
    client: httpx.AsyncClient,
    channel: str,
    path: str,
    *,
    params: Optional[dict] = None,
    describe: Optional[Callable[[dict], str]] = None,
) -> ChannelHealth:
    """Perform a lightweight authenticated GET and report channel reachability."""
    try:
        response = await client.get(path, params=params)
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        body = error.response.text.strip().replace("\n", " ")
        return ChannelHealth(
            channel=channel,
            ok=False,
            detail=f"HTTP {error.response.status_code}: {body[:200]}",
        )
    except httpx.HTTPError as error:
        return ChannelHealth(channel=channel, ok=False, detail=str(error))

    data = response.json() if response.content else {}
    detail = describe(data) if describe else "ok"
    return ChannelHealth(channel=channel, ok=True, detail=detail)