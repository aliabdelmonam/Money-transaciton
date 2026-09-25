from dataclasses import dataclass, field

@dataclass(frozen=True)

class ChannelEvent:
    channel:str
    conversation_id:str
    provider_message_id: str | None = field(default=None, kw_only=True)