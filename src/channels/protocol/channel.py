from typing import Protocol, runtime_checkable

@runtime_checkable
class Channel(Protocol):
    name:str

    async def send()-> str|None:  ...

    async def close()-> None: ...

    async def download_media()-> bytes: ...

    