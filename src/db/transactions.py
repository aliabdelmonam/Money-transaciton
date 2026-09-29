"""Persist receipts: one ``transactions`` row per image a user sends."""

import hashlib
import re
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from channels.events.messages import ImageMessageReceived
from channels.models.media import InboundMedia
from db.models import ExtractionStatus, Transaction
from db.session import Database

# Without at least one of these the OCR found nothing that looks like a transaction.
KEY_FIELDS = ("amount", "total", "reference")
MONEY_FIELDS = ("amount", "fees", "total")
PARTY_COLUMNS = ("name", "phone", "email", "account", "bank")
# A real e-mail has a dotted domain; "name@instapay" is an InstaPay handle (an account).
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


class TransactionStore:
    """Write receipts to the database; every call is its own unit of work."""

    def __init__(self, database: Database):
        self._database = database

    async def record_received(self, event: ImageMessageReceived) -> Optional[int]:
        """Store a pending row for a receipt image as soon as it arrives.

        Returns the new row id, or ``None`` when this message is already
        stored: Meta redelivers webhooks it thinks were missed, and the
        caller should not OCR and answer the same image twice.
        """
        try:
            async with self._database.session() as session:
                if event.provider_message_id and await _find(session, event):
                    return None
                row = _new_row(event)
                session.add(row)
                await session.flush()
                return row.id
        except IntegrityError:
            # A concurrent delivery of the same message won the insert.
            return None

    async def record_result(self, event: ImageMessageReceived, media: InboundMedia, result: dict) -> int:
        """Store the image and what the extractor read from it."""
        async with self._database.session() as session:
            row = await _find_or_add(session, event)
            _set_image(row, media)
            apply_result(row, result)
            await session.flush()
            return row.id

    async def record_failure(
        self, event: ImageMessageReceived, error: str, media: Optional[InboundMedia] = None
    ) -> int:
        """Mark the receipt as failed (download or OCR), keeping the image if we have it."""
        async with self._database.session() as session:
            row = await _find_or_add(session, event)
            if media is not None:
                _set_image(row, media)
            row.extraction_status = ExtractionStatus.FAILED
            row.error = error
            await session.flush()
            return row.id


def apply_result(row: Transaction, result: dict) -> None:
    """Copy an extractor result onto ``row``; fields with no column go to ``other_data``."""
    found = any(result.get(key) is not None for key in KEY_FIELDS)
    row.extraction_status = ExtractionStatus.EXTRACTED if found else ExtractionStatus.NO_DATA
    row.error = None
    row.provider = result.get("provider")
    row.transfer_status = result.get("status")
    row.transfer_type = result.get("type")
    for key in MONEY_FIELDS:
        value = result.get(key)
        setattr(row, key, None if value is None else Decimal(str(value)))
    row.currency = result.get("currency")
    row.reference_id = result.get("reference")
    row.date = _parse_datetime(result.get("datetime"))
    row.notes = result.get("note")

    other: dict[str, Any] = {}
    for side in ("sender", "receiver"):
        party = dict(result.get(side) or {})
        account = party.get("account")
        if isinstance(account, str) and EMAIL.fullmatch(account):
            party["email"] = party.pop("account")
        for column in PARTY_COLUMNS:
            setattr(row, f"{side}_{column}", party.pop(column, None))
        if party:                                   # via, other, ...
            other[side] = party
    unmapped = [item for item in result.get("unmapped") or [] if item.get("type")]
    if unmapped:                                    # typed leftovers: extra dates, money, phones
        other["unmapped"] = unmapped
    row.other_data = other or None
    row.raw_result = result


def _new_row(event: ImageMessageReceived) -> Transaction:
    attachment = event.attachment
    return Transaction(
        channel=event.channel,
        message_id=event.provider_message_id,
        user_phone=event.sender.id,
        user_name=event.sender.name,
        received_at=_parse_datetime((attachment.metadata or {}).get("timestamp")),
        caption=attachment.caption,
        image_mime_type=attachment.mime_type,
        image_filename=attachment.filename,
        image_size=attachment.size,
        extraction_status=ExtractionStatus.PENDING,
    )


def _set_image(row: Transaction, media: InboundMedia) -> None:
    row.image = media.data
    row.image_size = len(media.data)
    row.image_sha256 = hashlib.sha256(media.data).hexdigest()
    row.image_mime_type = row.image_mime_type or media.mime_type
    row.image_filename = row.image_filename or media.filename


async def _find(session: AsyncSession, event: ImageMessageReceived) -> Optional[Transaction]:
    return await session.scalar(
        select(Transaction).where(
            Transaction.channel == event.channel, Transaction.message_id == event.provider_message_id
        )
    )


async def _find_or_add(session: AsyncSession, event: ImageMessageReceived) -> Transaction:
    """The row stored on arrival; created here if that first write failed."""
    row = await _find(session, event) if event.provider_message_id else None
    if row is None:
        row = _new_row(event)
        session.add(row)
    return row


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value) if value else None
    except (TypeError, ValueError):
        return None
