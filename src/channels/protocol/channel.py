from typing import Protocol, runtime_checkable

from channels.models.outgoing import OutgoingMessage

@runtime_checkable
class Channel(Protocol):
    name:str

    async def send(self, message: OutgoingMessage)-> str|None:  ...

    async def close(self)-> None: ...

    async def download_media()-> bytes: ...

    