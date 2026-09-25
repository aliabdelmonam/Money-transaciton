from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AttachmentType(str, Enum):
    IMAGE = "image"
  


class Attachment(BaseModel):
    type: AttachmentType
    url: Optional[str] = None
    ref: Optional[str] = None
    caption: Optional[str] = None
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    size: Optional[int] = None
    data: Optional[bytes] = None
    metadata: Optional[dict] = Field(default=None)