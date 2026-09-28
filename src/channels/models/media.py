from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class InboundMedia:
    data: bytes
    mime_type: str
    filename: Optional[str] = None