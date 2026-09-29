from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import undefer

from channels.models.attachment import AttachmentType
from db.models import (
    Attachment,
    ExtractionStatus,
    Message,
    MessageDirection,
    MessageStatus,
    MessageType,
    PartyRole,
    Transaction,
    TransactionParty,
    User,
)


def inbound_image(user: User, provider_message_id: str = "wamid.1") -> Message:
    return Message(
        user=user,
        channel="whatsapp",
        conversation_id=user.external_id,
        direction=MessageDirection.INBOUND,
        type=MessageType.IMAGE,
        status=MessageStatus.RECEIVED,
        provider_message_id=provider_message_id,
    )


IMAGE = bytes([0x89]) + b'PNG' + bytes(range(256)) * 64   # arbitrary binary, not valid UTF-8


async def test_receipt_round_trip(database):
    """Every Receipt field we keep, plus the image itself, survives a save and reload."""
    async with database.session() as session:
        user = User(channel="whatsapp", external_id="201080719837", name="Mahmoud")
        attachment = Attachment(
            message=inbound_image(user),
            type=AttachmentType.IMAGE,
            provider_ref="media-1",
            mime_type="image/png",
            size_bytes=len(IMAGE),
            data=IMAGE,
        )
        session.add(
            Transaction(
                user=user,
                attachment=attachment,
                extraction_status=ExtractionStatus.EXTRACTED,
                provider="InstaPay",
                transfer_status="success",
                amount=Decimal("1000.00"),
                fees=Decimal("2.50"),
                total=Decimal("1002.50"),
                currency="EGP",
                reference="433891004642",
                occurred_at_raw="19 Sep 2026 01:37 PM",
                occurred_at=datetime(2026, 9, 19, 13, 37, tzinfo=timezone.utc),
                note="Living Expenses",
                other_data={"Transaction Type": "Transfer", "رقم العملية": "123"},
                raw_result={"ocr_lines": ["Transaction Successful", "1,000 EGP"]},
                parties=[
                    TransactionParty(
                        role=PartyRole.SENDER,
                        name="محمد عادل توفيق محمد",
                        phone="01012345678",
                        email="mohamed@example.com",
                        account="muhammedd002@instapay",
                    ),
                    TransactionParty(
                        role=PartyRole.RECEIVER,
                        name="Mahmoud F A*********",
                        phone="01080719837",
                        email="mahmoud@example.com",
                        account_type="Mobile Wallet",
                    ),
                ],
            )
        )

    async with database.session() as session:
        tx = (await session.scalars(select(Transaction))).one()
        assert tx.extraction_status is ExtractionStatus.EXTRACTED
        assert tx.transfer_status == "success"
        assert (tx.amount, tx.fees, tx.total) == (Decimal("1000.00"), Decimal("2.50"), Decimal("1002.50"))
        assert tx.reference == "433891004642"
        assert tx.occurred_at_raw == "19 Sep 2026 01:37 PM"
        assert tx.occurred_at.replace(tzinfo=None) == datetime(2026, 9, 19, 13, 37)
        assert tx.note == "Living Expenses"
        assert tx.other_data == {"Transaction Type": "Transfer", "رقم العملية": "123"}
        assert tx.raw_result["ocr_lines"][1] == "1,000 EGP"
        assert (tx.sender.name, tx.sender.phone, tx.sender.email) == (
            "محمد عادل توفيق محمد", "01012345678", "mohamed@example.com"
        )
        assert (tx.receiver.name, tx.receiver.phone, tx.receiver.email) == (
            "Mahmoud F A*********", "01080719837", "mahmoud@example.com"
        )
        assert tx.created_at is not None

        attachment = await session.scalar(
            select(Attachment).where(Attachment.id == tx.attachment_id).options(undefer(Attachment.data))
        )
        assert attachment.type is AttachmentType.IMAGE
        assert attachment.data == IMAGE


async def test_image_bytes_are_not_loaded_by_default(database):
    async with database.session() as session:
        user = User(channel="whatsapp", external_id="1")
        session.add(Attachment(message=inbound_image(user), type=AttachmentType.IMAGE, data=IMAGE))

    async with database.session() as session:
        attachment = await session.scalar(select(Attachment))
        assert "data" not in attachment.__dict__
        assert await attachment.awaitable_attrs.data == IMAGE


async def test_extraction_status_defaults_to_pending(database):
    async with database.session() as session:
        user = User(channel="whatsapp", external_id="1")
        session.add(Transaction(user=user))
    async with database.session() as session:
        tx = (await session.scalars(select(Transaction))).one()
        assert tx.extraction_status is ExtractionStatus.PENDING
        assert tx.parties == []
        assert tx.sender is None


async def test_duplicate_webhook_message_is_rejected(database):
    async with database.session() as session:
        session.add(inbound_image(User(channel="whatsapp", external_id="1"), "wamid.dup"))

    with pytest.raises(IntegrityError):
        async with database.session() as session:
            user = await session.scalar(select(User))
            session.add(inbound_image(user, "wamid.dup"))


async def test_outbound_messages_without_provider_id_are_allowed(database):
    async with database.session() as session:
        user = User(channel="whatsapp", external_id="1")
        for _ in range(2):
            session.add(
                Message(
                    user=user,
                    channel="whatsapp",
                    conversation_id="1",
                    direction=MessageDirection.OUTBOUND,
                    type=MessageType.TEXT,
                    status=MessageStatus.FAILED,
                    text="Transaction details",
                )
            )


async def test_deleting_user_cascades_in_the_database(database):
    """Checks that SQLite foreign keys are switched on, not just ORM cascades."""
    async with database.session() as session:
        user = User(channel="whatsapp", external_id="1")
        attachment = Attachment(message=inbound_image(user), type=AttachmentType.IMAGE)
        session.add(
            Transaction(
                user=user,
                attachment=attachment,
                parties=[TransactionParty(role=PartyRole.SENDER, name="Omar")],
            )
        )

    async with database.session() as session:
        await session.execute(delete(User))

    async with database.session() as session:
        for model in (Message, Attachment, Transaction, TransactionParty):
            assert await session.scalar(select(func.count()).select_from(model)) == 0
