"""OCR contract every engine adapter must satisfy."""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in pixels, top-left origin."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(1, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def scaled(self, factor: float) -> "BBox":
        return BBox(*(int(round(v * factor)) for v in (self.x1, self.y1, self.x2, self.y2)))


@dataclass(frozen=True)
class OCRResult:
    """One detected text line.

    `text` is the engine's best reading; `alts` holds every reading it produced
    (including `text`) so downstream matching can try all of them.
    """
    text: str
    confidence: float
    bbox: BBox
    engine: str = ""
    alts: tuple[str, ...] = field(default_factory=tuple)

    @property
    def readings(self) -> tuple[str, ...]:
        return self.alts or (self.text,)


@runtime_checkable
class OCR(Protocol):
    """Reads text lines from a BGR uint8 image."""

    name: str

    def read(self, image: np.ndarray) -> list[OCRResult]:
        ...
