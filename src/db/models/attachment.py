from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, LargeBinary, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from channels.models.attachment import AttachmentType
from db.base import Base, str_enum

if TYPE_CHECKING:
    from db.models.message import Message
    from db.models.transaction import Transaction


class Attachment(Base):
    """A media file carried by a message, e.g. the receipt screenshot.

    ``data`` holds the image bytes themselves. It is deferred, so listing
    attachments never loads the images; load it with
    ``options(undefer(Attachment.data))`` when you need them.
    """

    __tablename__ = "attachments"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"), index=True)
    type: Mapped[AttachmentType] = mapped_column(str_enum(AttachmentType))
    provider_ref: Mapped[Optional[str]] = mapped_column(String(128))   # WhatsApp media id
    mime_type: Mapped[Optional[str]] = mapped_column(String(64))
    filename: Mapped[Optional[str]] = mapped_column(String(255))
    size_bytes: Mapped[Optional[int]] = mapped_column(Integer)
    # Same screenshot sent twice -> same hash; indexed for duplicate-receipt checks.
    sha256: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    data: Mapped[Optional[bytes]] = mapped_column(LargeBinary, deferred=True)
    storage_path: Mapped[Optional[str]] = mapped_column(String(1024))   # copy on disk, if any
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    message: Mapped["Message"] = relationship(back_populates="attachments")
    transaction: Mapped[Optional["Transaction"]] = relationship(back_populates="attachment")

    def __repr__(self) -> str:
        return f"Attachment(id={self.id!r}, type={self.type.value!r}, filename={self.filename!r})"
