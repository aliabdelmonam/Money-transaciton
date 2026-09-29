import logging
from typing import Optional

import httpx

from channels.adapters.whatsapp.mapper import WhatsAppMessageMapper
from channels.models.exceptions import (
    AuthenticationError,
    ChannelError,
    MessageSendError,
    RateLimitError,
)
from channels.models.outgoing import OutgoingMessage, SentMessage

logger = logging.getLogger(__name__)


class WhatsAppSender:
    """Send outgoing messages through the WhatsApp Cloud API ``/messages`` endpoint.

    Free-form text is only delivered inside the 24-hour customer service
    window, i.e. as a reply to a user who messaged the bot recently.
    """

    def __init__(self, client: httpx.AsyncClient, mapper: WhatsAppMessageMapper):
        self._client = client
        self._mapper = mapper

    async def send(self, message: OutgoingMessage) -> list[SentMessage]:
        """Send the message, split into as many WhatsApp messages as the length limit needs.

        Returns one ``SentMessage`` per part, in order. If a part fails, the
        remaining parts are not sent and ``MessageSendError`` says which
        parts already went out.
        """
        requests = self._mapper.to_requests(message)
        sent: list[SentMessage] = []
        for index, request in enumerate(requests):
            try:
                message_id = await self._post(request.path, request.payload)
            except ChannelError as error:
                unsent = [r.text or "" for r in requests[index:]]
                raise MessageSendError(error, sent, unsent) from error
            sent.append(SentMessage(text=request.text or "", provider_message_id=message_id))
        return sent

    async def _post(self, path: str, payload: dict) -> Optional[str]:
        try:
            response = await self._client.post(path, json=payload)
        except httpx.HTTPError as error:
            raise ChannelError(f"whatsapp send failed: {error!r}") from error

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
