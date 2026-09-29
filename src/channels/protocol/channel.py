from typing import Optional, Protocol, runtime_checkable

from channels.models.attachment import Attachment
from channels.models.media import InboundMedia
from channels.models.outgoing import OutgoingMessage, SentMessage

@runtime_checkable
class Channel(Protocol):
    name:str

    async def send(self, message: OutgoingMessage) -> list[SentMessage]: ...

    async def close(self)-> None: ...

    async def fetch_media(self, attachment: Attachment)-> Optional[InboundMedia]: ...
