from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base, TimestampMixin, str_enum

if TYPE_CHECKING:
    from db.models.attachment import Attachment
    from db.models.user import User


class MessageDirection(str, Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class MessageType(str, Enum):
    TEXT = "text"
    IMAGE = "image"


class MessageStatus(str, Enum):
    RECEIVED = "received"       # inbound, stored
    SENT = "sent"               # outbound, accepted by the provider
    DELIVERED = "delivered"     # outbound, MessageDelivered event
    READ = "read"               # outbound, MessageRead event
    FAILED = "failed"


class Message(TimestampMixin, Base):
    """A message received from or sent to a user.

    ``(channel, provider_message_id)`` is unique, so a webhook delivery that
    Meta retries can be detected and skipped instead of being OCR'd twice.
    """

    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("channel", "provider_message_id"),
        Index("ix_messages_conversation_created", "channel", "conversation_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    channel: Mapped[str] = mapped_column(String(32))
    conversation_id: Mapped[str] = mapped_column(String(64))
    direction: Mapped[MessageDirection] = mapped_column(str_enum(MessageDirection))
    type: Mapped[MessageType] = mapped_column(str_enum(MessageType))
    status: Mapped[MessageStatus] = mapped_column(str_enum(MessageStatus))
    # None for an outbound message the provider rejected before assigning an id.
    provider_message_id: Mapped[Optional[str]] = mapped_column(String(128))
    reply_to_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL"), index=True
    )
    text: Mapped[Optional[str]] = mapped_column(Text)        # body, or the image caption
    error: Mapped[Optional[str]] = mapped_column(Text)
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))  # provider timestamp
    delivered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    read_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    user: Mapped["User"] = relationship(back_populates="messages")
    reply_to: Mapped[Optional["Message"]] = relationship(remote_side=[id])
    attachments: Mapped[list["Attachment"]] = relationship(
        back_populates="message", cascade="all, delete-orphan", passive_deletes=True
    )

    def __repr__(self) -> str:
        return (
            f"Message(id={self.id!r}, direction={self.direction.value!r}, "
            f"provider_message_id={self.provider_message_id!r})"
        )
