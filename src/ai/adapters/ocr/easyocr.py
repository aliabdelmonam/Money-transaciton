"""EasyOCR adapter: CRAFT detector + EasyOCR recognizer (Arabic + English)."""

import logging

import numpy as np

from src.config.ai import EasyOCRConfig
from ...protocols.ocr import OCRResult
from ...text.normalize import clean_text
from ._common import clip_boxes, make_result, merge_boxes, polygon_to_box

log = logging.getLogger(__name__)


class EasyOCRAdapter:
    name = "easyocr"

    def __init__(self, config: EasyOCRConfig | None = None, use_gpu: bool = True, merge_gap_ratio: float = 1.5):
        import easyocr

        self.cfg = config or EasyOCRConfig()
        self.merge_gap_ratio = merge_gap_ratio
        log.info("Loading EasyOCR (%s)", "+".join(self.cfg.langs))
        self._reader = easyocr.Reader(list(self.cfg.langs), gpu=use_gpu, verbose=False)

    def read(self, image: np.ndarray) -> list[OCRResult]:
        results = []
        for box in self._detect(image):
            text, conf = self._recognize(image, box)
            if conf < self.cfg.min_conf:
                continue
            res = make_result(clean_text(text), conf, box, self.name)
            if res:
                results.append(res)
        return results

    def _detect(self, image: np.ndarray) -> list[list[int]]:
        horizontal, free = self._reader.detect(image, width_ths=self.cfg.width_ths, add_margin=self.cfg.add_margin)
        boxes = [[int(x1), int(y1), int(x2), int(y2)] for x1, x2, y1, y2 in (horizontal[0] if horizontal else [])]
        boxes += [polygon_to_box(p) for p in (free[0] if free else [])]
        boxes = clip_boxes(boxes, image.shape)
        return merge_boxes(boxes, self.merge_gap_ratio) if self.cfg.merge_boxes else boxes

    def _recognize(self, image: np.ndarray, box: list[int]) -> tuple[str, float]:
        x1, y1, x2, y2 = box
        try:
            r = self._reader.recognize(image, horizontal_list=[[x1, x2, y1, y2]], free_list=[], detail=1)
        except Exception as e:   # one bad crop must not kill the whole image
            log.debug("EasyOCR recognize failed on %s: %s", box, e)
            return "", 0.0
        if not r:
            return "", 0.0
        return " ".join(x[1] for x in r), float(np.mean([x[2] for x in r]))
