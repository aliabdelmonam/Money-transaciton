from typing import TYPE_CHECKING, Optional

from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from db.models.message import Message
    from db.models.transaction import Transaction


class User(TimestampMixin, Base):
    """Someone who talks to the bot on a channel (a WhatsApp number, a Telegram id...)."""

    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("channel", "external_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    channel: Mapped[str] = mapped_column(String(32))
    external_id: Mapped[str] = mapped_column(String(64))    # WhatsApp wa_id, Telegram user id
    name: Mapped[Optional[str]] = mapped_column(String(255))
    username: Mapped[Optional[str]] = mapped_column(String(255))

    messages: Mapped[list["Message"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )
    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )

    def __repr__(self) -> str:
        return f"User(id={self.id!r}, channel={self.channel!r}, external_id={self.external_id!r})"
