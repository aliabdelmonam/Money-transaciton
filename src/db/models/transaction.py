from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base, Money, TimestampMixin, str_enum

if TYPE_CHECKING:
    from db.models.attachment import Attachment
    from db.models.user import User


class ExtractionStatus(str, Enum):
    PENDING = "pending"         # image stored, OCR not finished
    EXTRACTED = "extracted"     # found at least one key field (amount / total / reference)
    NO_DATA = "no_data"         # OCR ran but nothing looked like a transaction
    FAILED = "failed"           # OCR or parsing raised; see ``error``


class PartyRole(str, Enum):
    SENDER = "sender"
    RECEIVER = "receiver"


class Transaction(TimestampMixin, Base):
    """A money transfer read from a receipt screenshot.

    Typed columns hold the fields worth querying; ``raw_result`` keeps the
    full extractor output (evidence, OCR lines, warnings) for audits and
    for re-parsing when the extractor improves.
    """

    __tablename__ = "transactions"
    __table_args__ = (
        # Same receipt submitted twice (or by two users) -> same provider + reference.
        Index("ix_transactions_provider_reference", "provider", "reference"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    # One receipt per image. Nullable so a transaction can also be entered by hand.
    attachment_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("attachments.id", ondelete="SET NULL"), unique=True
    )
    extraction_status: Mapped[ExtractionStatus] = mapped_column(
        str_enum(ExtractionStatus), default=ExtractionStatus.PENDING, index=True
    )
    ocr_engine: Mapped[Optional[str]] = mapped_column(String(32))     # paddle | tesseract | easyocr

    provider: Mapped[Optional[str]] = mapped_column(String(64))       # InstaPay, Vodafone Cash, ...
    transfer_status: Mapped[Optional[str]] = mapped_column(String(32))  # success | failed, as printed
    transfer_type: Mapped[Optional[str]] = mapped_column(String(64))
    amount: Mapped[Optional[Decimal]] = mapped_column(Money)
    fees: Mapped[Optional[Decimal]] = mapped_column(Money)
    total: Mapped[Optional[Decimal]] = mapped_column(Money)
    currency: Mapped[Optional[str]] = mapped_column(String(3))        # ISO 4217, e.g. EGP
    reference: Mapped[Optional[str]] = mapped_column(String(128))
    occurred_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), index=True)
    occurred_at_raw: Mapped[Optional[str]] = mapped_column(String(64))  # date text as read
    note: Mapped[Optional[str]] = mapped_column(Text)
    other_data: Mapped[Optional[dict[str, Any]]]    # fields read that have no column (Receipt.other_data)

    raw_result: Mapped[Optional[dict[str, Any]]]
    error: Mapped[Optional[str]] = mapped_column(Text)

    user: Mapped["User"] = relationship(back_populates="transactions")
    attachment: Mapped[Optional["Attachment"]] = relationship(back_populates="transaction")
    parties: Mapped[list["TransactionParty"]] = relationship(
        back_populates="transaction",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )

    def party(self, role: PartyRole) -> Optional["TransactionParty"]:
        return next((p for p in self.parties if p.role == role), None)

    @property
    def sender(self) -> Optional["TransactionParty"]:
        return self.party(PartyRole.SENDER)

    @property
    def receiver(self) -> Optional["TransactionParty"]:
        return self.party(PartyRole.RECEIVER)

    def __repr__(self) -> str:
        return (
            f"Transaction(id={self.id!r}, provider={self.provider!r}, amount={self.amount!r}, "
            f"currency={self.currency!r}, reference={self.reference!r})"
        )


class TransactionParty(Base):
    """The sender or receiver side of a transaction.

    Covers both extractor outputs: ``account`` holds a bank account number,
    wallet number or InstaPay address (``wallet_id`` / ``account``), and
    ``account_type`` the channel it went through (``account_type`` / ``via``,
    e.g. "Mobile Wallet"). Anything else read for the party goes in ``extra``.
    """

    __tablename__ = "transaction_parties"
    __table_args__ = (UniqueConstraint("transaction_id", "role"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id", ondelete="CASCADE"))
    role: Mapped[PartyRole] = mapped_column(str_enum(PartyRole))
    name: Mapped[Optional[str]] = mapped_column(String(255))
    name_alt: Mapped[Optional[str]] = mapped_column(String(255))   # e.g. the Arabic spelling
    phone: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    email: Mapped[Optional[str]] = mapped_column(String(255))
    account: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    bank: Mapped[Optional[str]] = mapped_column(String(64))
    account_type: Mapped[Optional[str]] = mapped_column(String(64))
    extra: Mapped[Optional[dict[str, Any]]]

    transaction: Mapped["Transaction"] = relationship(back_populates="parties")

    def __repr__(self) -> str:
        return f"TransactionParty(role={self.role.value!r}, name={self.name!r}, account={self.account!r})"
