"""Image -> OCR -> structured receipt. The single entry point other layers should call."""

from dataclasses import dataclass

import numpy as np

from ..adapters.ocr import create_ocr
from src.config.ai import AIConfig
from ..extraction import LayoutReceiptExtractor, Receipt
from ..extraction.layout import Line, Token
from ..imaging import decode_image, load_image, resize_for_ocr
from ..protocols.ocr import OCR


@dataclass
class ReceiptAnalysis:
    receipt: Receipt
    lines: list[Line]
    tokens: list[Token]      # boxes in `image` coordinates
    image: np.ndarray        # the resized image OCR actually ran on
    scale: float             # image = original * scale

    def text_lines(self) -> list[str]:
        return [ln.text() for ln in self.lines]

    def to_dict(self) -> dict:
        out = self.receipt.to_dict()
        out["text_lines"] = self.text_lines()
        out["tokens"] = [{"text": t.text, "alts": list(t.alts), "conf": t.ocr.confidence,
                          "engine": t.ocr.engine, "role": t.role, "line": t.line,
                          "bbox": [int(round(v / self.scale)) for v in (t.x1, t.y1, t.x2, t.y2)]}
                         for t in self.tokens]
        return out


class ReceiptPipeline:
    def __init__(self, ocr: OCR | None = None, config: AIConfig | None = None):
        self.config = config or AIConfig()
        self.ocr = ocr or create_ocr(self.config)
        self.extractor = LayoutReceiptExtractor(self.config.extraction)

    def analyze(self, image: np.ndarray) -> ReceiptAnalysis:
        img, scale = resize_for_ocr(image, self.config.image)
        receipt, lines, tokens = self.extractor.parse(self.ocr.read(img))
        return ReceiptAnalysis(receipt, lines, tokens, img, scale)

    def analyze_file(self, path: str) -> ReceiptAnalysis:
        return self.analyze(load_image(path))

    def analyze_bytes(self, data: bytes) -> ReceiptAnalysis:
        return self.analyze(decode_image(data))
