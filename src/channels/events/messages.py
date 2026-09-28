from dataclasses import dataclass

from channels.events.base import ChannelEvent
from channels.models.attachment import Attachment
from channels.models.user import User

@dataclass(frozen=True)
class ImageMessageReceived(ChannelEvent):
    sender: User
    attachment: Attachment

@dataclass(frozen=True)
class MessageDelivered(ChannelEvent):
    message_id: str

@dataclass(frozen=True)
class MessageRead(ChannelEvent):
    message_id: str