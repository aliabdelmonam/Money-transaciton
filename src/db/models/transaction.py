from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from sqlalchemy import DateTime, Index, Integer, LargeBinary, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base, Money, TimestampMixin, str_enum


class ExtractionStatus(str, Enum):
    PENDING = "pending"         # image received, OCR not finished
    EXTRACTED = "extracted"     # found at least one key field (amount / total / reference)
    NO_DATA = "no_data"         # OCR ran but nothing looked like a transaction
    FAILED = "failed"           # download, OCR or parsing failed; see ``error``


class Transaction(TimestampMixin, Base):
    """One receipt image a user sent, and the transaction read from it.

    The only table: who sent the image and when (from the channel), the
    image itself, and the extracted fields. ``(channel, message_id)`` is
    unique, so a webhook delivery that Meta retries is detected and not
    OCR'd twice. ``other_data`` holds fields read that have no column;
    ``raw_result`` the full extractor output, for audits and re-parsing.
    """

    __tablename__ = "transactions"
    __table_args__ = (
        UniqueConstraint("channel", "message_id"),
        # Same receipt submitted twice (or by two users) -> same provider + reference.
        Index("ix_transactions_provider_reference", "provider", "reference_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    # ---- from the channel: who sent the image, and when
    channel: Mapped[str] = mapped_column(String(32))
    message_id: Mapped[Optional[str]] = mapped_column(String(128))     # WhatsApp wamid
    user_phone: Mapped[str] = mapped_column(String(64), index=True)    # WhatsApp wa_id of the user
    user_name: Mapped[Optional[str]] = mapped_column(String(255))      # their WhatsApp profile name
    received_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    caption: Mapped[Optional[str]] = mapped_column(Text)

    # ---- the image. Deferred, so listing transactions never loads the bytes;
    # load them with ``options(undefer(Transaction.image))``.
    image: Mapped[Optional[bytes]] = mapped_column(LargeBinary, deferred=True)
    image_mime_type: Mapped[Optional[str]] = mapped_column(String(64))
    image_filename: Mapped[Optional[str]] = mapped_column(String(255))
    image_size: Mapped[Optional[int]] = mapped_column(Integer)
    # Same screenshot sent twice -> same hash; indexed for duplicate-receipt checks.
    image_sha256: Mapped[Optional[str]] = mapped_column(String(64), index=True)

    # ---- read from the image
    extraction_status: Mapped[ExtractionStatus] = mapped_column(
        str_enum(ExtractionStatus), default=ExtractionStatus.PENDING, index=True
    )
    error: Mapped[Optional[str]] = mapped_column(Text)
    provider: Mapped[Optional[str]] = mapped_column(String(64))         # InstaPay, Vodafone Cash, ...
    transfer_status: Mapped[Optional[str]] = mapped_column(String(32))  # success | pending | failed
    transfer_type: Mapped[Optional[str]] = mapped_column(String(64))
    amount: Mapped[Optional[Decimal]] = mapped_column(Money)
    fees: Mapped[Optional[Decimal]] = mapped_column(Money)
    total: Mapped[Optional[Decimal]] = mapped_column(Money)
    currency: Mapped[Optional[str]] = mapped_column(String(3))          # ISO 4217, e.g. EGP
    reference_id: Mapped[Optional[str]] = mapped_column(String(128))
    date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), index=True)
    notes: Mapped[Optional[str]] = mapped_column(Text)

    sender_name: Mapped[Optional[str]] = mapped_column(String(255))
    sender_phone: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    sender_email: Mapped[Optional[str]] = mapped_column(String(255))
    sender_account: Mapped[Optional[str]] = mapped_column(String(128))  # bank account, IPA handle, card
    sender_bank: Mapped[Optional[str]] = mapped_column(String(64))

    receiver_name: Mapped[Optional[str]] = mapped_column(String(255))
    receiver_phone: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    receiver_email: Mapped[Optional[str]] = mapped_column(String(255))
    receiver_account: Mapped[Optional[str]] = mapped_column(String(128))
    receiver_bank: Mapped[Optional[str]] = mapped_column(String(64))

    other_data: Mapped[Optional[dict[str, Any]]]
    raw_result: Mapped[Optional[dict[str, Any]]]

    def __repr__(self) -> str:
        return (
            f"Transaction(id={self.id!r}, provider={self.provider!r}, amount={self.amount!r}, "
            f"currency={self.currency!r}, reference_id={self.reference_id!r})"
        )
