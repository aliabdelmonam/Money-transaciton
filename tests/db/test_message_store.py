import asyncio
import hashlib
from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import undefer

from channels.events.messages import ImageMessageReceived, MessageDelivered, MessageRead
from channels.models.attachment import Attachment as ChannelAttachment
from channels.models.attachment import AttachmentType
from channels.models.media import InboundMedia
from channels.models.outgoing import OutgoingMessage
from channels.models.user import User as ChannelUser
from db.messages import MessageStore
from db.models import Attachment, Message, MessageDirection, MessageStatus, MessageType, User

WA_ID = "201080719837"


def image_event(message_id: str = "wamid.in1", media_id: str = "media-1", name: str = "Mahmoud"):
    return ImageMessageReceived(
        channel="whatsapp",
        conversation_id=WA_ID,
        provider_message_id=message_id,
        sender=ChannelUser(id=WA_ID, name=name),
        attachment=ChannelAttachment(
            type=AttachmentType.IMAGE,
            ref=media_id,
            caption="my transfer",
            mime_type="image/jpeg",
            filename=f"{media_id}.jpg",
            metadata={"sha256": "provider-hash", "timestamp": "2026-09-29T10:00:00+00:00"},
        ),
    )


def reply(text: str = "*Transaction details*", reply_to: str | None = "wamid.in1") -> OutgoingMessage:
    return OutgoingMessage(conversation_id=WA_ID, text=text, reply_to_message_id=reply_to)


@pytest.fixture
def store(database) -> MessageStore:
    return MessageStore(database)


async def count(database, model) -> int:
    async with database.session() as session:
        return await session.scalar(select(func.count()).select_from(model))


async def test_record_inbound_stores_user_message_and_attachment(store, database):
    message_id = await store.record_inbound(image_event())
    assert message_id is not None

    async with database.session() as session:
        message = await session.get(Message, message_id)
        assert message.direction is MessageDirection.INBOUND
        assert message.type is MessageType.IMAGE
        assert message.status is MessageStatus.RECEIVED
        assert message.provider_message_id == "wamid.in1"
        assert message.text == "my transfer"
        assert message.sent_at.replace(tzinfo=None) == datetime(2026, 9, 29, 10, 0)

        user = await session.get(User, message.user_id)
        assert (user.channel, user.external_id, user.name) == ("whatsapp", WA_ID, "Mahmoud")

        attachment = await session.scalar(select(Attachment).where(Attachment.message_id == message_id))
        assert (attachment.provider_ref, attachment.mime_type, attachment.filename) == (
            "media-1", "image/jpeg", "media-1.jpg"
        )


async def test_redelivered_message_is_reported_as_duplicate(store, database):
    assert await store.record_inbound(image_event()) is not None
    assert await store.record_inbound(image_event()) is None
    assert await count(database, Message) == 1


async def test_concurrent_redeliveries_store_one_message(store, database):
    results = await asyncio.gather(*(store.record_inbound(image_event()) for _ in range(3)))
    assert sum(result is not None for result in results) == 1
    assert await count(database, Message) == 1


async def test_same_user_is_reused_and_profile_name_refreshed(store, database):
    await store.record_inbound(image_event("wamid.a", "m-a", name="Mahmoud"))
    await store.record_inbound(image_event("wamid.b", "m-b", name="Mahmoud F."))
    assert await count(database, User) == 1
    async with database.session() as session:
        assert (await session.scalar(select(User))).name == "Mahmoud F."


async def test_save_media_stores_the_image(store, database):
    event = image_event()
    await store.record_inbound(event)
    data = b"\xff\xd8\xff" + bytes(range(256)) * 10

    saved = await store.save_media("whatsapp", event.attachment, InboundMedia(data=data, mime_type="image/jpeg"))

    assert saved
    async with database.session() as session:
        attachment = await session.scalar(select(Attachment).options(undefer(Attachment.data)))
        assert attachment.data == data
        assert attachment.size_bytes == len(data)
        assert attachment.sha256 == hashlib.sha256(data).hexdigest()


async def test_save_media_for_unknown_attachment_is_a_no_op(store):
    attachment = ChannelAttachment(type=AttachmentType.IMAGE, ref="never-recorded")
    assert not await store.save_media("whatsapp", attachment, InboundMedia(data=b"x", mime_type="image/png"))


async def test_record_outbound_links_the_reply(store, database):
    inbound_id = await store.record_inbound(image_event())
    outbound_id = await store.record_outbound("whatsapp", reply(), "wamid.out1")

    async with database.session() as session:
        message = await session.get(Message, outbound_id)
        assert message.direction is MessageDirection.OUTBOUND
        assert message.type is MessageType.TEXT
        assert message.status is MessageStatus.SENT
        assert message.text == "*Transaction details*"
        assert message.reply_to_id == inbound_id
        assert message.sent_at is not None
    assert await count(database, User) == 1


async def test_record_failed_outbound(store, database):
    outbound_id = await store.record_outbound("whatsapp", reply(reply_to=None), None, error="HTTP 500")
    async with database.session() as session:
        message = await session.get(Message, outbound_id)
        assert message.status is MessageStatus.FAILED
        assert message.error == "HTTP 500"
        assert message.provider_message_id is None
        assert message.sent_at is None


async def test_delivery_receipts(store, database):
    outbound_id = await store.record_outbound("whatsapp", reply(), "wamid.out1")

    assert await store.record_status(MessageDelivered(channel="whatsapp", conversation_id=WA_ID, message_id="wamid.out1"))
    assert await store.record_status(MessageRead(channel="whatsapp", conversation_id=WA_ID, message_id="wamid.out1"))
    # A late "delivered" must not downgrade "read".
    assert await store.record_status(MessageDelivered(channel="whatsapp", conversation_id=WA_ID, message_id="wamid.out1"))

    async with database.session() as session:
        message = await session.get(Message, outbound_id)
        assert message.status is MessageStatus.READ
        assert message.delivered_at is not None
        assert message.read_at is not None


async def test_receipt_for_unknown_message(store):
    event = MessageRead(channel="whatsapp", conversation_id=WA_ID, message_id="wamid.unknown")
    assert not await store.record_status(event)


async def test_receipt_never_touches_inbound_messages(store, database):
    inbound_id = await store.record_inbound(image_event())
    event = MessageRead(channel="whatsapp", conversation_id=WA_ID, message_id="wamid.in1")
    assert not await store.record_status(event)
    async with database.session() as session:
        assert (await session.get(Message, inbound_id)).status is MessageStatus.RECEIVED
