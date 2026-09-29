import json

import httpx
import pytest

from channels.adapters.whatsapp.mapper import MAX_TEXT_LENGTH, WhatsAppMessageMapper
from channels.adapters.whatsapp.sender import WhatsAppSender
from channels.models.exceptions import MessageSendError, RateLimitError
from channels.models.outgoing import OutgoingMessage

LONG_TEXT = "\n".join(f"Line {n}: " + "x" * 90 for n in range(100))   # ~10k chars


def sender_with(handler) -> tuple[WhatsAppSender, list[dict]]:
    """A sender whose HTTP calls go to ``handler``; returns it and the payloads posted."""
    posted: list[dict] = []

    def record(request: httpx.Request) -> httpx.Response:
        posted.append(json.loads(request.content))
        return handler(len(posted))

    client = httpx.AsyncClient(transport=httpx.MockTransport(record), base_url="https://graph.test/v25.0/1")
    return WhatsAppSender(client, WhatsAppMessageMapper()), posted


def accepted(n: int) -> httpx.Response:
    return httpx.Response(200, json={"messages": [{"id": f"wamid.p{n}"}]})


def test_long_text_is_split_within_the_limit():
    requests = WhatsAppMessageMapper().to_requests(OutgoingMessage(conversation_id="2010", text=LONG_TEXT))
    texts = [request.text for request in requests]

    assert len(texts) == 3
    assert all(len(text) <= MAX_TEXT_LENGTH for text in texts)
    assert all(request.payload["text"]["body"] == request.text for request in requests)
    # Cut on line breaks: no line is split across two messages, none is lost.
    assert "\n".join(texts).split("\n") == LONG_TEXT.split("\n")


def test_short_text_is_one_message():
    requests = WhatsAppMessageMapper().to_requests(OutgoingMessage(conversation_id="2010", text="hi"))
    assert [request.text for request in requests] == ["hi"]


async def test_send_returns_every_part():
    sender, posted = sender_with(accepted)
    message = OutgoingMessage(conversation_id="2010", text=LONG_TEXT, reply_to_message_id="wamid.in1")

    sent = await sender.send(message)

    assert [part.provider_message_id for part in sent] == ["wamid.p1", "wamid.p2", "wamid.p3"]
    assert [part.text for part in sent] == [payload["text"]["body"] for payload in posted]
    # Only the first part quotes the user's message.
    assert [payload.get("context") for payload in posted] == [{"message_id": "wamid.in1"}, None, None]


async def test_failure_mid_way_reports_sent_and_unsent_parts():
    sender, posted = sender_with(lambda n: accepted(n) if n == 1 else httpx.Response(429, text="slow down"))

    with pytest.raises(MessageSendError) as caught:
        await sender.send(OutgoingMessage(conversation_id="2010", text=LONG_TEXT))

    error = caught.value
    assert len(posted) == 2                                    # part 3 was never attempted
    assert [part.provider_message_id for part in error.sent] == ["wamid.p1"]
    assert len(error.unsent) == 2
    assert "\n".join([error.sent[0].text, *error.unsent]).split("\n") == LONG_TEXT.split("\n")
    assert isinstance(error.error, RateLimitError)             # original cause kept
    assert "part 2 of 3" in str(error)


async def test_failure_of_a_single_message():
    sender, _ = sender_with(lambda n: httpx.Response(500, text="boom"))

    with pytest.raises(MessageSendError) as caught:
        await sender.send(OutgoingMessage(conversation_id="2010", text="hi"))

    assert caught.value.sent == []
    assert caught.value.unsent == ["hi"]
    assert "part" not in str(caught.value)
