"""PaddleOCR adapter.

One detector finds the text lines, then every line is read by two recognizers:
  - arabic_PP-OCRv5 : Arabic + Latin + digits (the only one that can read Arabic)
  - PP-OCRv6        : strongest on English / digits / emails
The best reading becomes the text, both are kept in `alts`.
"""

import logging
import os

import numpy as np

from src.config.ai import PaddleConfig
from ...protocols.ocr import OCRResult
from ...text.normalize import NON_AR_LATIN, clean_text, script_counts
from ._common import clip_boxes, crop, make_result, merge_boxes, polygon_to_box

log = logging.getLogger(__name__)


class PaddleOCRAdapter:
    name = "paddle"

    def __init__(self, config: PaddleConfig | None = None, use_gpu: bool = True, merge_gap_ratio: float = 1.5):
        self.cfg = config or PaddleConfig()
        self.merge_gap_ratio = merge_gap_ratio
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        # paddleocr must be imported before paddle: paddleocr pulls in torch, and on Windows
        # torch fails to load its DLLs (WinError 127) if paddle was imported first
        from paddleocr import TextDetection, TextRecognition
        import paddle

        device = "gpu:0" if use_gpu and paddle.device.is_compiled_with_cuda() else "cpu"
        if use_gpu and device == "cpu":
            log.warning("paddlepaddle has no CUDA build -> running PaddleOCR on CPU")
        # enable_mkldnn=False works around a oneDNN crash in paddlepaddle 3.3 on CPU
        common = {"device": device, "enable_mkldnn": False}
        log.info("Loading PaddleOCR (%s + %s + %s) on %s",
                 self.cfg.det_model, self.cfg.ar_rec_model, self.cfg.en_rec_model, device)
        self._det = TextDetection(model_name=self.cfg.det_model, **common)
        self._rec_ar = TextRecognition(model_name=self.cfg.ar_rec_model, **common)
        self._rec_en = (TextRecognition(model_name=self.cfg.en_rec_model, **common)
                        if self.cfg.en_rec_model else None)

    def read(self, image: np.ndarray) -> list[OCRResult]:
        boxes = self._detect(image)
        crops = [crop(image, b, max(2, int(self.cfg.crop_pad * (b[3] - b[1])))) for b in boxes]
        ar = self._recognize(self._rec_ar, crops)
        en = self._recognize(self._rec_en, crops)
        results = []
        for box, (a_txt, a_conf), (e_txt, e_conf) in zip(boxes, ar, en):
            text, conf, engine = self._choose(a_txt, a_conf, e_txt, e_conf)
            if conf < self.cfg.min_conf:
                continue
            res = make_result(text, conf, box, engine, (a_txt, e_txt))
            if res:
                results.append(res)
        return results

    def _detect(self, image: np.ndarray) -> list[list[int]]:
        res = self._det.predict(image)
        boxes = [polygon_to_box(p) for p in (res[0]["dt_polys"] if res else [])]
        boxes = clip_boxes(boxes, image.shape)
        return merge_boxes(boxes, self.merge_gap_ratio) if self.cfg.merge_boxes else boxes

    def _recognize(self, model, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        if model is None or not crops:
            return [("", 0.0)] * len(crops)
        out = model.predict(crops, batch_size=self.cfg.batch_size)
        return [(str(r["rec_text"]), float(r["rec_score"])) for r in out]

    def _choose(self, a_txt: str, a_conf: float, e_txt: str, e_conf: float) -> tuple[str, float, str]:
        a_txt, e_txt = clean_text(a_txt), clean_text(e_txt)
        # the latin model turns Arabic into junk like 'äenl li' / 'ス' / '☆'
        if not e_txt or NON_AR_LATIN.search(e_txt):
            return a_txt, a_conf, "paddle_ar"
        if not a_txt:
            return e_txt, e_conf, "paddle_en"
        if script_counts(a_txt)[0]:                   # Arabic seen -> only the Arabic model is trustworthy
            return a_txt, a_conf, "paddle_ar"
        if e_conf >= a_conf - self.cfg.en_margin:     # pure latin/digits -> prefer the latin model
            return e_txt, e_conf, "paddle_en"
        return a_txt, a_conf, "paddle_ar"
