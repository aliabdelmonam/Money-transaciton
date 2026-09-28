"""Helpers shared by the OCR adapters."""

import numpy as np

from ...protocols.ocr import BBox, OCRResult
from ...text.normalize import NON_AR_LATIN, clean_text, has_word_char

Box = list[int]   # [x1, y1, x2, y2]


def clip_boxes(boxes: list[Box], shape: tuple[int, ...], min_size: int = 4) -> list[Box]:
    h, w = shape[:2]
    boxes = [[max(0, a), max(0, b), min(w, c), min(h, d)] for a, b, c, d in boxes]
    return [b for b in boxes if b[2] - b[0] > min_size and b[3] - b[1] > min_size]


def polygon_to_box(poly) -> Box:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _should_merge(a: Box, b: Box, gap_ratio: float) -> bool:
    ha, hb = a[3] - a[1], b[3] - b[1]
    if min(ha, hb) <= 0:
        return False
    ov = min(a[3], b[3]) - max(a[1], b[1])
    if ov <= 0 or ov / min(ha, hb) < 0.6 or min(ha, hb) / max(ha, hb) < 0.45:
        return False
    gap = max(a[0], b[0]) - min(a[2], b[2])
    return gap <= gap_ratio * max(ha, hb)


def merge_boxes(boxes: list[Box], gap_ratio: float) -> list[Box]:
    """Merge word boxes that sit on the same line close to each other."""
    changed = True
    while changed:
        changed = False
        pending = sorted(boxes, key=lambda b: (b[1], b[0]))
        out = []
        while pending:
            a = pending.pop(0)
            i = 0
            while i < len(pending):
                if _should_merge(a, pending[i], gap_ratio):
                    b = pending.pop(i)
                    a = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    changed = True
                else:
                    i += 1
            out.append(a)
        boxes = out
    return boxes


def crop(img: np.ndarray, box: Box, pad: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    return img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]


def make_result(text: str, conf: float, box: Box, engine: str, readings=()) -> OCRResult | None:
    """Build an OCRResult with deduplicated, junk-free alternatives; None if nothing usable."""
    alts: list[str] = []
    for a in (text, *(clean_text(r) for r in readings)):
        if a and a not in alts and not NON_AR_LATIN.search(a):
            alts.append(a)
    if not alts or not any(has_word_char(a) for a in alts):
        return None
    return OCRResult(text=text, confidence=round(float(conf), 3), bbox=BBox(*map(int, box)),
                     engine=engine, alts=tuple(alts))
