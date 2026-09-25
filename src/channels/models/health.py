from pydantic import BaseModel


class ChannelHealth(BaseModel):
    channel: str
    ok: bool
    detail: str