import logging
from typing import Mapping, Optional, Protocol, runtime_checkable

from channels.events.base import ChannelEvent
from channels.events.messages import ImageMessageReceived
from channels.models.outgoing import OutgoingMessage
from channels.protocol.channel import Channel

logger = logging.getLogger(__name__)

# Events sent by a user that deserve a reply; status updates (delivered/read) do not.
REPLYABLE_EVENTS: tuple[type[ChannelEvent], ...] = (ImageMessageReceived,)


@runtime_checkable
class ReplyProvider(Protocol):
    """Decide what text to answer an event with; ``None`` means no reply."""

    async def reply(self, event: ChannelEvent) -> Optional[str]: ...


class StaticReplyProvider:
    """Reply with a predefined message chosen by event type."""

    def __init__(
        self,
        replies: Optional[Mapping[type[ChannelEvent], str]] = None,
        default: Optional[str] = None,
    ):
        self._replies = dict(replies or {})
        self._default = default

    async def reply(self, event: ChannelEvent) -> Optional[str]:
        for event_type, text in self._replies.items():
            if isinstance(event, event_type):
                return text
        return self._default


class FallbackReplyProvider:
    """Ask each provider in order and use the first non-empty reply.

    A provider that raises is logged and skipped, so an AI provider can be
    placed first with a ``StaticReplyProvider`` behind it as a safety net.
    """

    def __init__(self, *providers: ReplyProvider):
        self._providers = providers

    async def reply(self, event: ChannelEvent) -> Optional[str]:
        for provider in self._providers:
            try:
                text = await provider.reply(event)
            except Exception:
                logger.exception("reply provider %s failed", type(provider).__name__)
                continue
            if text:
                return text
        return None


class ChatbotResponder:
    """Answer inbound user messages on a channel using a reply provider."""

    def __init__(self, channel: Channel, provider: ReplyProvider):
        self._channel = channel
        self._provider = provider

    async def handle(self, event: Optional[ChannelEvent]) -> Optional[str]:
        """Reply to ``event`` if it is a user message; returns the sent message id."""
        if not isinstance(event, REPLYABLE_EVENTS):
            return None
        text = await self._provider.reply(event)
        if not text:
            return None
        return await self.send_text(
            event.conversation_id, text, reply_to=event.provider_message_id
        )

    async def send_text(
        self, conversation_id: str, text: str, reply_to: Optional[str] = None
    ) -> Optional[str]:
        """Send an arbitrary text message, e.g. a result computed later."""
        return await self._channel.send(
            OutgoingMessage(
                conversation_id=conversation_id,
                text=text,
                reply_to_message_id=reply_to,
            )
        )
