from typing import Optional

from pydantic import BaseModel


class OutgoingMessage(BaseModel):
    conversation_id: str
    text: str
    reply_to_message_id: Optional[str] = None
    preview_url: bool = False


class SentMessage(BaseModel):
    """One message the provider accepted.

    A long ``OutgoingMessage`` is split into several of these, each with
    its own provider id (and its own delivered/read receipts).
    """

    text: str
    provider_message_id: Optional[str] = None
