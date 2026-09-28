import logging
from typing import Optional

import httpx

from channels.adapters.whatsapp.mapper import WhatsAppMessageMapper
from channels.models.exceptions import AuthenticationError, ChannelError, RateLimitError
from channels.models.outgoing import OutgoingMessage

logger = logging.getLogger(__name__)


class WhatsAppSender:
    """Send outgoing messages through the WhatsApp Cloud API ``/messages`` endpoint.

    Free-form text is only delivered inside the 24-hour customer service
    window, i.e. as a reply to a user who messaged the bot recently.
    """

    def __init__(self, client: httpx.AsyncClient, mapper: WhatsAppMessageMapper):
        self._client = client
        self._mapper = mapper

    async def send(self, message: OutgoingMessage) -> Optional[str]:
        """Send the message and return the provider id of the last part sent."""
        message_id = None
        for request in self._mapper.to_requests(message):
            message_id = await self._post(request.path, request.payload)
        return message_id

    async def _post(self, path: str, payload: dict) -> Optional[str]:
        try:
            response = await self._client.post(path, json=payload)
        except httpx.HTTPError as error:
            raise ChannelError(f"whatsapp send failed: {error}") from error

        if response.status_code == 429:
            raise RateLimitError(f"whatsapp rate limited: {response.text[:200]}")
        if response.status_code in (401, 403):
            raise AuthenticationError(f"whatsapp auth failed: {response.text[:200]}")
        if response.is_error:
            raise ChannelError(
                f"whatsapp send failed: HTTP {response.status_code}: {response.text[:200]}"
            )

        messages = response.json().get("messages") or []
        message_id = messages[0].get("id") if messages else None
        logger.info("whatsapp: sent message %s to %s", message_id, payload.get("to"))
        return message_id
