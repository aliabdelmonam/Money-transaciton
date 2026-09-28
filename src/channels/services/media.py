import logging

import httpx

from channels.models.exceptions import ChannelError, MediaTooLargeError

logger = logging.getLogger(__name__)

DEFAULT_MAX_BYTES = 20 * 1024 * 1024 # 20 MB


class MediaService:
    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES, timeout: float = 30.0):
        self._max_bytes = max_bytes
        self._timeout = timeout

    async def download(self, url: str) -> bytes:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
        except httpx.HTTPError as error:
            logger.error("media download failed: %s", error)
            raise ChannelError(f"media download failed: {url}") from error
        self._ensure_size(len(response.content), url)
        return response.content

    def _ensure_size(self, size: int, url: str) -> None:
        if size > self._max_bytes:
            raise MediaTooLargeError(f"media exceeds {self._max_bytes} bytes: {url}")