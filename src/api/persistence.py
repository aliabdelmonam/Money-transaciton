"""Glue between the channel flow and the transaction store.

Recording never breaks the bot: a failed database write is logged and the
message is still handled / sent.
"""
import logging
from collections.abc import Awaitable
from typing import Optional

from channels.events.base import ChannelEvent
from channels.events.messages import ImageMessageReceived
from channels.models.outgoing import SentMessage
from channels.services.reply import ChatbotResponder
from db.transactions import TransactionStore

logger = logging.getLogger(__name__)


class RecordingResponder:
    """Store each receipt image before answering it, and skip redeliveries."""

    def __init__(self, responder: ChatbotResponder, store: TransactionStore):
        self._responder = responder
        self._store = store

    async def handle(self, event: Optional[ChannelEvent]) -> list[SentMessage]:
        if isinstance(event, ImageMessageReceived):
            try:
                if await self._store.record_received(event) is None:
                    logger.info("skipping redelivered message %s", event.provider_message_id)
                    return []
            except Exception:
                logger.exception("failed to store inbound message %s; answering anyway", event.provider_message_id)
        return await self._responder.handle(event)


async def record(what: str, write: Awaitable) -> None:
    try:
        await write
    except Exception:
        logger.exception("failed to store %s", what)
