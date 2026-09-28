"""OCR engine adapters. Engines are imported lazily so only the one you use must be installed."""

from src.config.ai import AIConfig
from ...protocols.ocr import OCR

ENGINES = ("paddle", "tesseract", "easyocr")


def create_ocr(config: AIConfig | None = None) -> OCR:
    cfg = config or AIConfig()
    if cfg.engine == "paddle":
        from .paddle import PaddleOCRAdapter
        return PaddleOCRAdapter(cfg.paddle, cfg.use_gpu, cfg.merge_gap_ratio)
    if cfg.engine == "tesseract":
        from .tesseract import TesseractOCRAdapter
        return TesseractOCRAdapter(cfg.tesseract)
    if cfg.engine == "easyocr":
        from .easyocr import EasyOCRAdapter
        return EasyOCRAdapter(cfg.easyocr, cfg.use_gpu, cfg.merge_gap_ratio)
    raise ValueError(f"unknown OCR engine {cfg.engine!r}, expected one of {ENGINES}")


__all__ = ["ENGINES", "create_ocr"]
