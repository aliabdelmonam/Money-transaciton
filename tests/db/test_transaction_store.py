import hashlib
from datetime import datetime
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import undefer

from channels.events.messages import ImageMessageReceived
from channels.models.attachment import Attachment, AttachmentType
from channels.models.media import InboundMedia
from channels.models.user import User
from db.models import ExtractionStatus, Transaction
from db.transactions import TransactionStore

MEDIA = InboundMedia(data=b"\xff\xd8image-bytes", mime_type="image/jpeg", filename="1113.jpg")

RESULT = {
    "provider": "InstaPay", "status": "success", "type": None,
    "amount": 1000, "fees": 2.5, "total": 1002.5, "currency": "EGP",
    "sender": {"name": "Ali Abdelmonam", "account": "ali@example.com", "via": "InstaPay"},
    "receiver": {"name": "Mahmoud F A***", "phone": "01080719837", "account": "mahmoud@instapay",
                 "other": ["01001496550"]},
    "reference": "433891004642", "datetime": "2026-09-19T13:37:00", "note": "Living Expenses",
    "evidence": [], "unmapped": [{"text": "Success!", "type": None},
                                 {"text": "20 Sep 2026", "type": "datetime"}],
}


def image_event(message_id="wamid.in1") -> ImageMessageReceived:
    return ImageMessageReceived(
        channel="whatsapp",
        conversation_id="201143846150",
        provider_message_id=message_id,
        sender=User(id="201143846150", name="Ali Abdelmonam"),
        attachment=Attachment(
            type=AttachmentType.IMAGE, ref="media-1", caption="my receipt", mime_type="image/jpeg",
            metadata={"timestamp": "2026-09-29T14:53:57+00:00"},
        ),
    )


@pytest.fixture
def store(database) -> TransactionStore:
    return TransactionStore(database)


async def load(database) -> Transaction:
    async with database.session() as session:
        return (await session.scalars(select(Transaction).options(undefer(Transaction.image)))).one()


async def test_received_image_is_stored_as_pending(store, database):
    assert await store.record_received(image_event()) is not None
    tx = await load(database)
    assert tx.extraction_status is ExtractionStatus.PENDING
    assert (tx.channel, tx.message_id, tx.user_phone, tx.user_name) == (
        "whatsapp", "wamid.in1", "201143846150", "Ali Abdelmonam"
    )
    assert tx.caption == "my receipt"
    assert tx.received_at.replace(tzinfo=None) == datetime(2026, 9, 29, 14, 53, 57)
    assert tx.image is None


async def test_redelivered_image_is_reported_as_duplicate(store, database):
    assert await store.record_received(image_event()) is not None
    assert await store.record_received(image_event()) is None
    async with database.session() as session:
        assert await session.scalar(select(func.count()).select_from(Transaction)) == 1


async def test_result_fills_the_pending_row(store, database):
    row_id = await store.record_received(image_event())
    assert await store.record_result(image_event(), MEDIA, RESULT) == row_id

    tx = await load(database)
    assert tx.extraction_status is ExtractionStatus.EXTRACTED
    assert tx.image == MEDIA.data
    assert tx.image_size == len(MEDIA.data)
    assert tx.image_sha256 == hashlib.sha256(MEDIA.data).hexdigest()
    assert tx.image_filename == "1113.jpg"
    assert (tx.provider, tx.transfer_status) == ("InstaPay", "success")
    assert (tx.amount, tx.fees, tx.total, tx.currency) == (
        Decimal("1000.00"), Decimal("2.50"), Decimal("1002.50"), "EGP"
    )
    assert tx.reference_id == "433891004642"
    assert tx.date.replace(tzinfo=None) == datetime(2026, 9, 19, 13, 37)
    assert tx.notes == "Living Expenses"
    # a real e-mail goes to *_email; an InstaPay handle stays an account
    assert (tx.sender_name, tx.sender_email, tx.sender_account) == ("Ali Abdelmonam", "ali@example.com", None)
    assert (tx.receiver_name, tx.receiver_phone, tx.receiver_email, tx.receiver_account) == (
        "Mahmoud F A***", "01080719837", None, "mahmoud@instapay"
    )
    assert tx.other_data == {
        "sender": {"via": "InstaPay"},
        "receiver": {"other": ["01001496550"]},
        "unmapped": [{"text": "20 Sep 2026", "type": "datetime"}],
    }
    assert tx.raw_result == RESULT


async def test_nothing_found_is_stored_as_no_data(store, database):
    await store.record_received(image_event())
    await store.record_result(image_event(), MEDIA, {"status": "success", "unmapped": []})
    tx = await load(database)
    assert tx.extraction_status is ExtractionStatus.NO_DATA
    assert tx.image == MEDIA.data
    assert tx.other_data is None


async def test_result_is_stored_even_if_the_pending_row_is_missing(store, database):
    await store.record_result(image_event(), MEDIA, RESULT)
    tx = await load(database)
    assert (tx.message_id, tx.amount) == ("wamid.in1", Decimal("1000.00"))


async def test_failure_keeps_the_image_and_the_error(store, database):
    await store.record_received(image_event())
    await store.record_failure(image_event(), "RuntimeError('ocr crashed')", MEDIA)
    tx = await load(database)
    assert tx.extraction_status is ExtractionStatus.FAILED
    assert tx.error == "RuntimeError('ocr crashed')"
    assert tx.image == MEDIA.data


async def test_failed_download_has_no_image(store, database):
    await store.record_received(image_event())
    await store.record_failure(image_event(), "could not download media")
    tx = await load(database)
    assert (tx.extraction_status, tx.image) == (ExtractionStatus.FAILED, None)
