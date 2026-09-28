from typing import Optional

from pydantic import BaseModel


class OutgoingMessage(BaseModel):
    conversation_id: str
    text: str
    reply_to_message_id: Optional[str] = None
    preview_url: bool = False
