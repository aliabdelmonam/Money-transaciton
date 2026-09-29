"""Persist channel messages: inbound images, bot replies and their delivery receipts."""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional, Union

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from channels.events.messages import ImageMessageReceived, MessageDelivered, MessageRead
from channels.models.attachment import Attachment as ChannelAttachment
from channels.models.media import InboundMedia
from channels.models.outgoing import OutgoingMessage
from db.models import Attachment, Message, MessageDirection, MessageStatus, MessageType, User
from db.session import Database

logger = logging.getLogger(__name__)

StatusEvent = Union[MessageDelivered, MessageRead]


class MessageStore:
    """Write channel traffic to the database; every call is its own unit of work."""

    def __init__(self, database: Database):
        self._database = database

    async def record_inbound(self, event: ImageMessageReceived) -> Optional[int]:
        """Store an inbound image message with its sender and attachment metadata.

        Returns the new message id, or ``None`` when this provider message id
        is already stored: Meta redelivers webhooks it thinks were missed, and
        the caller should not OCR and answer the same image twice.
        """
        for attempt in range(2):
            try:
                return await self._insert_inbound(event)
            except IntegrityError:
                # Either a concurrent delivery of the same message won the insert
                # (-> duplicate), or two first messages raced to create the user (-> retry).
                if await self._inbound_exists(event):
                    return None
                if attempt:
                    raise
        return None

    async def save_media(
        self, channel: str, attachment: ChannelAttachment, media: InboundMedia
    ) -> bool:
        """Store the downloaded bytes on the attachment row recorded for ``attachment.ref``."""
        if not attachment.ref:
            return False
        async with self._database.session() as session:
            row = await session.scalar(
                select(Attachment)
                .join(Message)
                .where(Message.channel == channel, Attachment.provider_ref == attachment.ref)
                .order_by(Attachment.id.desc())
                .limit(1)
            )
            if row is None:
                logger.warning("no stored attachment for %s media %s", channel, attachment.ref)
                return False
            row.data = media.data
            row.size_bytes = len(media.data)
            row.sha256 = hashlib.sha256(media.data).hexdigest()
            row.mime_type = row.mime_type or media.mime_type
            row.filename = row.filename or media.filename
            return True

    async def record_outbound(
        self,
        channel: str,
        message: OutgoingMessage,
        provider_message_id: Optional[str],
        error: Optional[str] = None,
    ) -> int:
        """Store a message the bot sent (or tried to send, when ``error`` is set)."""
        async with self._database.session() as session:
            user = await _upsert_user(session, channel, message.conversation_id)
            reply_to = None
            if message.reply_to_message_id:
                reply_to = await _find_message(session, channel, message.reply_to_message_id)
            row = Message(
                user=user,
                channel=channel,
                conversation_id=message.conversation_id,
                direction=MessageDirection.OUTBOUND,
                type=MessageType.TEXT,
                status=MessageStatus.FAILED if error else MessageStatus.SENT,
                provider_message_id=provider_message_id,
                reply_to=reply_to,
                text=message.text,
                error=error,
                sent_at=None if error else _now(),
            )
            session.add(row)
            await session.flush()
            return row.id

    async def record_status(self, event: StatusEvent) -> bool:
        """Apply a delivered/read receipt to the outbound message it refers to.

        Receipts can arrive out of order, so a late "delivered" never
        downgrades a message that is already "read".
        """
        async with self._database.session() as session:
            row = await _find_message(session, event.channel, event.message_id)
            if row is None or row.direction is not MessageDirection.OUTBOUND:
                return False
            now = _now()
            row.delivered_at = row.delivered_at or now
            if isinstance(event, MessageRead):
                row.read_at = row.read_at or now
                row.status = MessageStatus.READ
            elif row.status != MessageStatus.READ:
                row.status = MessageStatus.DELIVERED
            return True

    async def _insert_inbound(self, event: ImageMessageReceived) -> Optional[int]:
        async with self._database.session() as session:
            if event.provider_message_id and await _find_message(
                session, event.channel, event.provider_message_id
            ):
                return None
            sender = event.sender
            user = await _upsert_user(session, event.channel, sender.id, sender.name, sender.username)
            attachment = event.attachment
            message = Message(
                user=user,
                channel=event.channel,
                conversation_id=event.conversation_id,
                direction=MessageDirection.INBOUND,
                type=MessageType.IMAGE,
                status=MessageStatus.RECEIVED,
                provider_message_id=event.provider_message_id,
                text=attachment.caption,
                sent_at=_parse_timestamp((attachment.metadata or {}).get("timestamp")),
                attachments=[
                    Attachment(
                        type=attachment.type,
                        provider_ref=attachment.ref,
                        mime_type=attachment.mime_type,
                        filename=attachment.filename,
                        size_bytes=attachment.size,
                    )
                ],
            )
            session.add(message)
            await session.flush()
            return message.id

    async def _inbound_exists(self, event: ImageMessageReceived) -> bool:
        if not event.provider_message_id:
            return False
        async with self._database.session() as session:
            return await _find_message(session, event.channel, event.provider_message_id) is not None


async def _find_message(session: AsyncSession, channel: str, provider_message_id: str) -> Optional[Message]:
    return await session.scalar(
        select(Message).where(
            Message.channel == channel, Message.provider_message_id == provider_message_id
        )
    )


async def _upsert_user(
    session: AsyncSession,
    channel: str,
    external_id: str,
    name: Optional[str] = None,
    username: Optional[str] = None,
) -> User:
    """Find the user by channel id, creating it or refreshing its profile name."""
    user = await session.scalar(
        select(User).where(User.channel == channel, User.external_id == external_id)
    )
    if user is None:
        user = User(channel=channel, external_id=external_id)
        session.add(user)
    if name:
        user.name = name
    if username:
        user.username = username
    return user


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value) if value else None
    except ValueError:
        return None


def _now() -> datetime:
    return datetime.now(timezone.utc)
