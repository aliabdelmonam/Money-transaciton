from typing import Optional

from pydantic import BaseModel


class ChannelInfo(BaseModel):
    name: str
    account_id: Optional[str] = None