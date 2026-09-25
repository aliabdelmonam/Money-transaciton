import logging
from typing import Optional

import httpx

from channels.adapters.whatsapp.mapper import WhatsAppMessageMapper
from channels.events.base import ChannelEvent
from channels.models.attachment import Attachment
from channels.models.capabilities import Capabilities, Capability
from channels.models.exceptions import MediaTooLargeError
from channels.models.health import ChannelHealth
from channels.models.media import InboundMedia
from channels.registry import register_channel
from channels.services.health import http_probe
from channels.services.media import DEFAULT_MAX_BYTES
from channels.services.webhook import WebhookService

logger = logging.getLogger(__name__)


@register_channel("whatsapp")
class WhatsappAdapter:
    """Inbound-only WhatsApp Cloud API adapter that accepts image messages.

    Nothing is sent back to the user; the adapter only verifies webhooks,
    parses incoming images into events and downloads their bytes.
    """

    name = "whatsapp"
    mapper_class = WhatsAppMessageMapper
    webhook_scheme = "hmac_sha256"
    capabilities = Capabilities(
        Capability.IMAGE,
    )

    def __init__(
        self,
        webhook: WebhookService,
        client: httpx.AsyncClient,
        config,
        max_media_bytes: int = DEFAULT_MAX_BYTES,
    ):
        self._webhook = webhook
        self._client = client
        self._verify_token = config.verify_token
        self._api_root = f"{config.api_base}/{config.api_version}"
        self._max_media_bytes = max_media_bytes
        self._client.headers["Authorization"] = f"Bearer {config.access_token}"

    @staticmethod
    def base_url(config) -> str:
        return f"{config.api_base}/{config.api_version}/{config.phone_number_id}"

    async def fetch_media(self, attachment: Attachment) -> Optional[InboundMedia]:
        """Download inbound media via WhatsApp's two-step flow.

        1. ``GET /{media_id}`` → JSON with ``url`` field
        2. ``GET {url}`` with Bearer token → binary data
        """
        if not attachment.ref:
            return None
        info = await self._resolve_media(attachment.ref)
        if not info or not info.get("url"):
            return None
        data = await self._download_media(info["url"])
        if not data:
            return None
        return InboundMedia(
            data=data,
            mime_type=attachment.mime_type or info.get("mime_type") or "application/octet-stream",
            filename=attachment.filename or "media",
        )

    async def _resolve_media(self, media_id: str) -> Optional[dict]:
        """Retrieve the metadata (temporary ``url``, ``mime_type``, ``file_size``)."""
        try:
            response = await self._client.get(f"{self._api_root}/{media_id}")
            response.raise_for_status()
        except httpx.HTTPError as error:
            logger.error("whatsapp media resolve failed for %s: %s", media_id, error)
            return None
        info = response.json()
        size = info.get("file_size")
        if size is not None and int(size) > self._max_media_bytes:
            raise MediaTooLargeError(
                f"whatsapp media {media_id} is {size} bytes (max {self._max_media_bytes})"
            )
        return info

    async def _download_media(self, url: str) -> Optional[bytes]:
        """Download binary content from the temporary WhatsApp media URL."""
        try:
            response = await self._client.get(url, timeout=30.0)
            response.raise_for_status()
        except httpx.HTTPError as error:
            logger.error("whatsapp media download failed: %s", error)
            return None
        if len(response.content) > self._max_media_bytes:
            raise MediaTooLargeError(
                f"whatsapp media exceeds {self._max_media_bytes} bytes"
            )
        return response.content

    async def health_check(self) -> ChannelHealth:
        return await http_probe(
            self._client,
            self.name,
            "",
            params={"fields": "display_phone_number,verified_name"},
            describe=lambda d: d.get("display_phone_number")
            or d.get("verified_name")
            or d.get("id")
            or "ok",
        )

    def verify(self, body: bytes, signature: Optional[str]) -> bool:
        """Check the ``X-Hub-Signature-256`` header against the app secret."""
        return self._webhook.verify(body, signature)

    def parse(self, payload: dict) -> Optional[ChannelEvent]:
        return self._webhook.parse(payload)

    def verify_challenge(
        self, mode: Optional[str], verify_token: Optional[str]
    ) -> bool:
        return (
            mode == "subscribe"
            and self._verify_token is not None
            and verify_token == self._verify_token
        )

    async def close(self) -> None:
        await self._client.aclose()
