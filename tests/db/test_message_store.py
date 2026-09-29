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
from channels.models.outgoing import OutgoingMessage, SentMessage
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


def parts(*ids: str) -> list[SentMessage]:
    return [SentMessage(text=f"part {n}", provider_message_id=id_) for n, id_ in enumerate(ids, 1)]


async def outbound_rows(database) -> list[Message]:
    async with database.session() as session:
        return list(await session.scalars(
            select(Message).where(Message.direction == MessageDirection.OUTBOUND).order_by(Message.id)
        ))


async def test_record_outbound_links_the_reply(store, database):
    inbound_id = await store.record_inbound(image_event())
    sent = [SentMessage(text="*Transaction details*", provider_message_id="wamid.out1")]
    [outbound_id] = await store.record_outbound("whatsapp", reply(), sent)

    async with database.session() as session:
        message = await session.get(Message, outbound_id)
        assert message.direction is MessageDirection.OUTBOUND
        assert message.type is MessageType.TEXT
        assert message.status is MessageStatus.SENT
        assert message.provider_message_id == "wamid.out1"
        assert message.text == "*Transaction details*"
        assert message.reply_to_id == inbound_id
        assert (message.part_index, message.part_of_id) == (0, None)
        assert message.sent_at is not None
    assert await count(database, User) == 1


async def test_split_reply_stores_one_row_per_whatsapp_message(store, database):
    inbound_id = await store.record_inbound(image_event())
    ids = await store.record_outbound("whatsapp", reply(), parts("wamid.p1", "wamid.p2", "wamid.p3"))

    rows = await outbound_rows(database)
    assert [row.id for row in rows] == ids
    assert [row.provider_message_id for row in rows] == ["wamid.p1", "wamid.p2", "wamid.p3"]
    assert [row.text for row in rows] == ["part 1", "part 2", "part 3"]
    assert [row.part_index for row in rows] == [0, 1, 2]
    assert [row.part_of_id for row in rows] == [None, ids[0], ids[0]]
    assert [row.reply_to_id for row in rows] == [inbound_id, None, None]   # WhatsApp quotes on part 1 only
    assert all(row.status is MessageStatus.SENT for row in rows)

    async with database.session() as session:
        first = await session.get(Message, ids[0])
        assert [part.id for part in await first.awaitable_attrs.parts] == ids[1:]


async def test_receipt_for_any_part_of_a_split_reply_is_applied(store, database):
    ids = await store.record_outbound("whatsapp", reply(), parts("wamid.p1", "wamid.p2", "wamid.p3"))

    assert await store.record_status(MessageRead(channel="whatsapp", conversation_id=WA_ID, message_id="wamid.p2"))

    async with database.session() as session:
        assert (await session.get(Message, ids[1])).status is MessageStatus.READ
        assert (await session.get(Message, ids[0])).status is MessageStatus.SENT


async def test_partly_sent_reply_keeps_the_sent_parts(store, database):
    ids = await store.record_outbound(
        "whatsapp", reply(), parts("wamid.p1"), unsent=["part 2", "part 3"], error="HTTP 500"
    )

    rows = await outbound_rows(database)
    assert [row.id for row in rows] == ids
    assert [row.status for row in rows] == [MessageStatus.SENT, MessageStatus.FAILED, MessageStatus.FAILED]
    assert [row.provider_message_id for row in rows] == ["wamid.p1", None, None]
    assert [row.text for row in rows] == ["part 1", "part 2", "part 3"]
    assert [row.error for row in rows] == [None, "HTTP 500", "HTTP 500"]
    assert [row.part_of_id for row in rows] == [None, ids[0], ids[0]]


async def test_record_failed_outbound(store, database):
    [outbound_id] = await store.record_outbound(
        "whatsapp", reply(reply_to=None), [], unsent=["*Transaction details*"], error="HTTP 500"
    )
    async with database.session() as session:
        message = await session.get(Message, outbound_id)
        assert message.status is MessageStatus.FAILED
        assert message.error == "HTTP 500"
        assert message.provider_message_id is None
        assert message.sent_at is None


async def test_nothing_sent_stores_nothing(store, database):
    assert await store.record_outbound("whatsapp", reply(), []) == []
    assert await count(database, Message) == 0


async def test_delivery_receipts(store, database):
    [outbound_id] = await store.record_outbound("whatsapp", reply(), parts("wamid.out1"))

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
