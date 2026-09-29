"""Glue between the channel flow and the message store.

Recording never breaks the bot: a failed database write is logged and the
message is still handled / sent.
"""
import logging
from collections.abc import Awaitable
from typing import Any, Optional

from channels.events.base import ChannelEvent
from channels.events.messages import ImageMessageReceived, MessageDelivered, MessageRead
from channels.models.attachment import Attachment
from channels.models.exceptions import MessageSendError
from channels.models.media import InboundMedia
from channels.models.outgoing import OutgoingMessage, SentMessage
from channels.protocol.channel import Channel
from channels.services.reply import ChatbotResponder
from db.messages import MessageStore

logger = logging.getLogger(__name__)


class RecordingChannel:
    """A channel that stores every message it sends and every image it downloads.

    Everything else (``verify``, ``parse``, ``health_check``, ...) goes
    straight to the wrapped adapter.
    """

    def __init__(self, channel: Channel, store: MessageStore):
        self._channel = channel
        self._store = store
        self.name = channel.name

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._channel, attr)

    async def send(self, message: OutgoingMessage) -> list[SentMessage]:
        try:
            sent = await self._channel.send(message)
        except MessageSendError as error:
            # Parts sent before the failure are real WhatsApp messages: keep them.
            await self._record_outbound(message, error.sent, error.unsent, str(error.error))
            raise
        except Exception as error:
            await self._record_outbound(message, [], [message.text], str(error))
            raise
        await self._record_outbound(message, sent)
        return sent

    async def _record_outbound(self, message: OutgoingMessage, sent, unsent=(), error=None) -> None:
        await _record("outbound message", self._store.record_outbound(self.name, message, sent, unsent, error))

    async def fetch_media(self, attachment: Attachment) -> Optional[InboundMedia]:
        media = await self._channel.fetch_media(attachment)
        if media is not None:
            await _record("media", self._store.save_media(self.name, attachment, media))
        return media

    async def close(self) -> None:
        await self._channel.close()


class RecordingResponder:
    """Store each webhook event before answering it, and skip redeliveries."""

    def __init__(self, responder: ChatbotResponder, store: MessageStore):
        self._responder = responder
        self._store = store

    async def handle(self, event: Optional[ChannelEvent]) -> list[SentMessage]:
        if isinstance(event, (MessageDelivered, MessageRead)):
            await _record("delivery status", self._store.record_status(event))
            return []
        if isinstance(event, ImageMessageReceived):
            try:
                if await self._store.record_inbound(event) is None:
                    logger.info("skipping redelivered message %s", event.provider_message_id)
                    return []
            except Exception:
                logger.exception("failed to store inbound message %s; answering anyway", event.provider_message_id)
        return await self._responder.handle(event)


async def _record(what: str, write: Awaitable) -> None:
    try:
        await write
    except Exception:
        logger.exception("failed to store %s", what)
