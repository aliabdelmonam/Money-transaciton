from typing import Optional

import pytest

from api.persistence import RecordingResponder
from api.receipts import TransactionReplyProvider
from channels.events.messages import ImageMessageReceived, MessageDelivered
from channels.models.attachment import Attachment, AttachmentType
from channels.models.media import InboundMedia
from channels.models.outgoing import SentMessage
from channels.models.user import User

MEDIA = InboundMedia(data=b"image-bytes", mime_type="image/jpeg")
RESULT = {"amount": 9, "currency": "EGP", "receiver": {"phone": "01001496550"}}


class FakeChannel:
    name = "whatsapp"

    def __init__(self, media: Optional[InboundMedia] = MEDIA):
        self.media = media

    async def fetch_media(self, attachment: Attachment) -> Optional[InboundMedia]:
        return self.media


class FakeReader:
    def __init__(self, result: Optional[dict] = None, error: Optional[Exception] = None):
        self.result = result if result is not None else RESULT
        self.error = error

    async def read(self, data: bytes, filename: str) -> dict:
        if self.error:
            raise self.error
        return self.result


class FakeStore:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[tuple] = []
        self.seen: set[str] = set()

    async def _call(self, *call):
        self.calls.append(call)
        if self.fail:
            raise RuntimeError("database is locked")

    async def record_received(self, event):
        await self._call("received", event.provider_message_id)
        if event.provider_message_id in self.seen:
            return None
        self.seen.add(event.provider_message_id)
        return len(self.seen)

    async def record_result(self, event, media, result):
        await self._call("result", event.provider_message_id, media.data, result)

    async def record_failure(self, event, error, media=None):
        await self._call("failure", event.provider_message_id, error, media and media.data)


class FakeResponder:
    def __init__(self):
        self.handled = []

    async def handle(self, event):
        self.handled.append(event)
        return [SentMessage(text="reply", provider_message_id="wamid.reply")]


def image_event(message_id: str = "wamid.in1") -> ImageMessageReceived:
    return ImageMessageReceived(
        channel="whatsapp",
        conversation_id="2010",
        provider_message_id=message_id,
        sender=User(id="2010"),
        attachment=Attachment(type=AttachmentType.IMAGE, ref="media-1"),
    )


async def test_responder_skips_redelivered_messages():
    store, inner = FakeStore(), FakeResponder()
    responder = RecordingResponder(inner, store)
    assert len(await responder.handle(image_event())) == 1
    assert await responder.handle(image_event()) == []
    assert len(inner.handled) == 1


async def test_responder_stores_nothing_for_status_updates():
    store, inner = FakeStore(), FakeResponder()
    event = MessageDelivered(channel="whatsapp", conversation_id="2010", message_id="wamid.out1")
    await RecordingResponder(inner, store).handle(event)
    assert store.calls == []


async def test_responder_still_answers_when_the_database_fails():
    inner = FakeResponder()
    assert len(await RecordingResponder(inner, FakeStore(fail=True)).handle(image_event())) == 1
    assert len(inner.handled) == 1


async def test_provider_stores_the_image_and_result():
    store = FakeStore()
    text = await TransactionReplyProvider(FakeChannel(), FakeReader(), store).reply(image_event())
    assert "Amount: 9 EGP" in text
    assert store.calls == [("result", "wamid.in1", b"image-bytes", RESULT)]


async def test_provider_still_replies_when_the_database_fails():
    text = await TransactionReplyProvider(FakeChannel(), FakeReader(), FakeStore(fail=True)).reply(image_event())
    assert "Amount: 9 EGP" in text


async def test_provider_records_a_failed_download():
    store = FakeStore()
    assert await TransactionReplyProvider(FakeChannel(media=None), FakeReader(), store).reply(image_event()) is None
    assert store.calls == [("failure", "wamid.in1", "could not download media", None)]


async def test_provider_records_an_ocr_error_and_reraises():
    store = FakeStore()
    provider = TransactionReplyProvider(FakeChannel(), FakeReader(error=RuntimeError("ocr crashed")), store)
    with pytest.raises(RuntimeError):
        await provider.reply(image_event())
    assert store.calls == [("failure", "wamid.in1", "RuntimeError('ocr crashed')", b"image-bytes")]
