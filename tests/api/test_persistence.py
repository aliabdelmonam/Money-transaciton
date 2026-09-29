from typing import Optional

import pytest

from api.persistence import RecordingChannel, RecordingResponder
from channels.events.messages import ImageMessageReceived, MessageDelivered
from channels.models.attachment import Attachment, AttachmentType
from channels.models.exceptions import ChannelError
from channels.models.media import InboundMedia
from channels.models.outgoing import OutgoingMessage
from channels.models.user import User

MEDIA = InboundMedia(data=b"image-bytes", mime_type="image/jpeg")


class FakeChannel:
    name = "whatsapp"

    def __init__(self, fail_send: bool = False):
        self.fail_send = fail_send
        self.sent: list[OutgoingMessage] = []

    async def send(self, message: OutgoingMessage) -> Optional[str]:
        if self.fail_send:
            raise ChannelError("HTTP 500")
        self.sent.append(message)
        return f"wamid.out{len(self.sent)}"

    async def fetch_media(self, attachment: Attachment) -> Optional[InboundMedia]:
        return MEDIA

    def verify(self, body: bytes, signature: Optional[str]) -> bool:
        return True

    async def close(self) -> None:
        pass


class FakeStore:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[tuple] = []
        self.seen: set[str] = set()

    async def _call(self, *call):
        self.calls.append(call)
        if self.fail:
            raise RuntimeError("database is locked")

    async def record_inbound(self, event):
        await self._call("inbound", event.provider_message_id)
        if event.provider_message_id in self.seen:
            return None
        self.seen.add(event.provider_message_id)
        return len(self.seen)

    async def save_media(self, channel, attachment, media):
        await self._call("media", channel, attachment.ref, media.data)

    async def record_outbound(self, channel, message, provider_message_id, error=None):
        await self._call("outbound", channel, message.text, provider_message_id, error)

    async def record_status(self, event):
        await self._call("status", event.message_id)


class FakeResponder:
    def __init__(self):
        self.handled = []

    async def handle(self, event):
        self.handled.append(event)
        return "wamid.reply"


def image_event(message_id: str = "wamid.in1") -> ImageMessageReceived:
    return ImageMessageReceived(
        channel="whatsapp",
        conversation_id="2010",
        provider_message_id=message_id,
        sender=User(id="2010"),
        attachment=Attachment(type=AttachmentType.IMAGE, ref="media-1"),
    )


async def test_channel_records_sent_messages():
    store = FakeStore()
    channel = RecordingChannel(FakeChannel(), store)
    assert await channel.send(OutgoingMessage(conversation_id="2010", text="hi")) == "wamid.out1"
    assert store.calls == [("outbound", "whatsapp", "hi", "wamid.out1", None)]


async def test_channel_records_failed_send_and_reraises():
    store = FakeStore()
    channel = RecordingChannel(FakeChannel(fail_send=True), store)
    with pytest.raises(ChannelError):
        await channel.send(OutgoingMessage(conversation_id="2010", text="hi"))
    assert store.calls == [("outbound", "whatsapp", "hi", None, "HTTP 500")]


async def test_channel_saves_downloaded_media():
    store = FakeStore()
    channel = RecordingChannel(FakeChannel(), store)
    attachment = Attachment(type=AttachmentType.IMAGE, ref="media-1")
    assert await channel.fetch_media(attachment) is MEDIA
    assert store.calls == [("media", "whatsapp", "media-1", b"image-bytes")]


async def test_channel_keeps_working_when_the_database_fails():
    inner = FakeChannel()
    channel = RecordingChannel(inner, FakeStore(fail=True))
    assert await channel.send(OutgoingMessage(conversation_id="2010", text="hi")) == "wamid.out1"
    assert await channel.fetch_media(Attachment(type=AttachmentType.IMAGE, ref="m")) is MEDIA
    assert len(inner.sent) == 1


async def test_channel_forwards_adapter_methods():
    assert RecordingChannel(FakeChannel(), FakeStore()).verify(b"", None) is True


async def test_responder_skips_redelivered_messages():
    store, inner = FakeStore(), FakeResponder()
    responder = RecordingResponder(inner, store)
    assert await responder.handle(image_event()) == "wamid.reply"
    assert await responder.handle(image_event()) is None
    assert len(inner.handled) == 1


async def test_responder_records_status_without_answering():
    store, inner = FakeStore(), FakeResponder()
    event = MessageDelivered(channel="whatsapp", conversation_id="2010", message_id="wamid.out1")
    assert await RecordingResponder(inner, store).handle(event) is None
    assert store.calls == [("status", "wamid.out1")]
    assert inner.handled == []


async def test_responder_still_answers_when_the_database_fails():
    inner = FakeResponder()
    assert await RecordingResponder(inner, FakeStore(fail=True)).handle(image_event()) == "wamid.reply"
    assert len(inner.handled) == 1
