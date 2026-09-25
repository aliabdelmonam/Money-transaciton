from typing import List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel

from channels.events.base import ChannelEvent
# from channels.models.outgoing import OutgoingMessage


class SendRequest(BaseModel):
    path: str
    payload: dict
    files: dict | None = None


@runtime_checkable
class MessageMapper(Protocol):
    def to_event(self, payload: dict) -> Optional[ChannelEvent]: ...

    # def to_requests(self, message: OutgoingMessage) -> List[SendRequest]: ...