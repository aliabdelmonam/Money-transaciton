"""All tunables for the AI layer. Override by constructing the dataclasses yourself."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PaddleConfig:
    det_model: str = "PP-OCRv6_medium_det"            # text line detector
    ar_rec_model: str = "arabic_PP-OCRv5_mobile_rec"  # Arabic + English + digits
    en_rec_model: Optional[str] = "PP-OCRv6_medium_rec"  # English / digits (None = Arabic only)
    min_conf: float = 0.30              # drop readings below this (0-1)
    en_margin: float = 0.05             # latin-only line: English model wins unless this much less confident
    crop_pad: float = 0.15              # padding around each line crop (ratio of box height)
    batch_size: int = 16
    merge_boxes: bool = False           # Paddle already returns whole lines


@dataclass
class TesseractConfig:
    cmd: Optional[str] = None           # e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe" (None = on PATH)
    tessdata_dir: Optional[str] = None  # folder with ara/eng.traineddata (tessdata_best)
    langs: tuple[str, ...] = ("ara", "eng")
    psm: int = 11                       # sparse text: receipts are scattered label/value pairs
    min_conf: float = 30.0              # 0-100


@dataclass
class EasyOCRConfig:
    langs: tuple[str, ...] = ("ar", "en")
    min_conf: float = 0.30
    width_ths: float = 0.9
    add_margin: float = 0.08
    merge_boxes: bool = True


@dataclass
class ImageConfig:
    target_long_side: int = 1600        # small screenshots are upscaled to this
    max_long_side: int = 2600           # huge photos are downscaled to this


@dataclass
class ExtractionConfig:
    line_y_tolerance: float = 0.6       # same line if |dy| <= tol * height
    fuzzy_label_threshold: float = 82   # fuzzy match score for labels (OCR typos)
    max_party_lines: int = 4            # max lines in a sender/receiver block
    party_gap_ratio: float = 3.5        # vertical gap bigger than this * height ends a block


@dataclass
class AIConfig:
    engine: str = "paddle"              # paddle | tesseract | easyocr
    use_gpu: bool = True
    merge_gap_ratio: float = 1.5        # merge boxes on a line if gap <= ratio * box height
    paddle: PaddleConfig = field(default_factory=PaddleConfig)
    tesseract: TesseractConfig = field(default_factory=TesseractConfig)
    easyocr: EasyOCRConfig = field(default_factory=EasyOCRConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
