"""Contract for turning OCR output into structured receipt data."""

from typing import Protocol, runtime_checkable

from ..extraction.models import Receipt
from .ocr import OCRResult


@runtime_checkable
class ReceiptExtractor(Protocol):
    def extract(self, results: list[OCRResult]) -> Receipt:
        ...
