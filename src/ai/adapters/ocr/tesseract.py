"""Tesseract adapter: page-level `image_to_data`, words grouped into lines."""

import logging
import os

import cv2
import numpy as np

from src.config.ai import TesseractConfig
from ...protocols.ocr import OCRResult
from ...text.normalize import clean_text, script_counts
from ._common import make_result

log = logging.getLogger(__name__)


class TesseractOCRAdapter:
    name = "tesseract"

    def __init__(self, config: TesseractConfig | None = None):
        import pytesseract

        self.cfg = config or TesseractConfig()
        self._tess = pytesseract
        if self.cfg.cmd:
            pytesseract.pytesseract.tesseract_cmd = self.cfg.cmd
        if self.cfg.tessdata_dir:
            os.environ["TESSDATA_PREFIX"] = os.path.abspath(self.cfg.tessdata_dir)
        available = set(pytesseract.get_languages(config=""))
        langs = [l for l in self.cfg.langs if l in available]
        if not langs:
            raise RuntimeError(f"none of {self.cfg.langs} installed for Tesseract (found {sorted(available)})")
        if "ara" in self.cfg.langs and "ara" not in available:
            log.warning("ara.traineddata not found -> Arabic accuracy will drop. Set tessdata_dir.")
        self._lang = "+".join(langs)
        self._config = f"--oem 1 --psm {self.cfg.psm}"

    def read(self, image: np.ndarray) -> list[OCRResult]:
        data = self._tess.image_to_data(self._prepare(image), lang=self._lang, config=self._config,
                                        output_type=self._tess.Output.DICT)
        lines: dict[tuple, list[int]] = {}
        for i, word in enumerate(data["text"]):
            if str(word).strip() and float(data["conf"][i]) >= 0:
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                lines.setdefault(key, []).append(i)
        results = []
        for idxs in lines.values():
            res = self._line_result(data, idxs)
            if res:
                results.append(res)
        return results

    def _line_result(self, data: dict, idxs: list[int]) -> OCRResult | None:
        conf = sum(float(data["conf"][i]) for i in idxs) / len(idxs)
        if conf < self.cfg.min_conf:
            return None
        ar, la = script_counts(" ".join(str(data["text"][i]) for i in idxs))
        words = sorted(idxs, key=lambda i: data["left"][i], reverse=ar > la)   # RTL for Arabic
        text = clean_text(" ".join(str(data["text"][i]) for i in words))
        box = [min(data["left"][i] for i in idxs), min(data["top"][i] for i in idxs),
               max(data["left"][i] + data["width"][i] for i in idxs),
               max(data["top"][i] + data["height"][i] for i in idxs)]
        return make_result(text, conf / 100, box, self.name)

    @staticmethod
    def _prepare(image: np.ndarray) -> np.ndarray:
        g = image.min(axis=2)          # coloured text (orange EGP) becomes dark
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
        if np.mean(g) < 127:           # dark mode screenshot
            g = 255 - g
        return g
