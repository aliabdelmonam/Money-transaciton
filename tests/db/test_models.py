from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import undefer

from db.models import ExtractionStatus, Transaction

IMAGE = bytes([0x89]) + b'PNG' + bytes(range(256)) * 64   # arbitrary binary, not valid UTF-8


def receipt(message_id="wamid.1", **fields) -> Transaction:
    return Transaction(channel="whatsapp", message_id=message_id, user_phone="201080719837", **fields)


async def test_receipt_round_trip(database):
    """Every column, the image included, survives a save and reload."""
    async with database.session() as session:
        session.add(
            receipt(
                user_name="Mahmoud",
                received_at=datetime(2026, 9, 19, 13, 40, tzinfo=timezone.utc),
                image=IMAGE,
                image_mime_type="image/png",
                image_size=len(IMAGE),
                extraction_status=ExtractionStatus.EXTRACTED,
                provider="InstaPay",
                transfer_status="success",
                amount=Decimal("1000.00"),
                fees=Decimal("2.50"),
                total=Decimal("1002.50"),
                currency="EGP",
                reference_id="433891004642",
                date=datetime(2026, 9, 19, 13, 37, tzinfo=timezone.utc),
                notes="Living Expenses",
                sender_name="محمد عادل توفيق محمد",
                sender_phone="01012345678",
                sender_email="mohamed@example.com",
                sender_account="muhammedd002@instapay",
                receiver_name="Mahmoud F A*********",
                receiver_phone="01080719837",
                receiver_email="mahmoud@example.com",
                other_data={"receiver": {"via": "Mobile Wallet"}},
                raw_result={"amount": 1000, "unmapped": []},
            )
        )

    async with database.session() as session:
        tx = await session.scalar(select(Transaction).options(undefer(Transaction.image)))
        assert (tx.channel, tx.message_id, tx.user_phone, tx.user_name) == (
            "whatsapp", "wamid.1", "201080719837", "Mahmoud"
        )
        assert tx.image == IMAGE
        assert tx.extraction_status is ExtractionStatus.EXTRACTED
        assert (tx.amount, tx.fees, tx.total) == (Decimal("1000.00"), Decimal("2.50"), Decimal("1002.50"))
        assert tx.reference_id == "433891004642"
        assert tx.date.replace(tzinfo=None) == datetime(2026, 9, 19, 13, 37)
        assert tx.notes == "Living Expenses"
        assert (tx.sender_name, tx.sender_phone, tx.sender_email) == (
            "محمد عادل توفيق محمد", "01012345678", "mohamed@example.com"
        )
        assert (tx.receiver_name, tx.receiver_phone, tx.receiver_email) == (
            "Mahmoud F A*********", "01080719837", "mahmoud@example.com"
        )
        assert tx.other_data == {"receiver": {"via": "Mobile Wallet"}}
        assert tx.raw_result["amount"] == 1000
        assert tx.created_at is not None


async def test_image_bytes_are_not_loaded_by_default(database):
    async with database.session() as session:
        session.add(receipt(image=IMAGE))

    async with database.session() as session:
        tx = await session.scalar(select(Transaction))
        assert "image" not in tx.__dict__
        assert await tx.awaitable_attrs.image == IMAGE


async def test_extraction_status_defaults_to_pending(database):
    async with database.session() as session:
        session.add(receipt())
    async with database.session() as session:
        tx = await session.scalar(select(Transaction))
        assert tx.extraction_status is ExtractionStatus.PENDING
        assert tx.amount is None and tx.sender_name is None


async def test_duplicate_webhook_message_is_rejected(database):
    async with database.session() as session:
        session.add(receipt("wamid.dup"))

    with pytest.raises(IntegrityError):
        async with database.session() as session:
            session.add(receipt("wamid.dup"))


async def test_rows_without_message_id_are_allowed(database):
    async with database.session() as session:
        session.add_all([receipt(None), receipt(None)])
