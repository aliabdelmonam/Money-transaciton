"""PaddleOCR: receipt lines with their boxes, the same interface as tesseract_ocr.

  1. image.prepare (shared with the Tesseract engine): dark mode / photo lighting / margins
     evened out, text scaled to ~28 px tall, tilt removed
  2. PP-OCRv6_medium_det finds the text lines; boxes of one line that overlap ("750" +
     "0 EGP") are merged. No textline orientation model: screenshots are upright, and it
     turned whole lines upside down ("muhammedd002@instapay" -> "Kedetsu!ozooppewweynw")
  3. every line is read by the Arabic and the English recognizer; the more plausible
     reading wins, a mixed line takes each script from its own model (recognize.py)
  4. masked names are rebuilt around their asterisks, counted by shape (masked.py)

Runs on the GPU when paddlepaddle is a CUDA build and sees one, else on the CPU.
"""
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
logging.disable(logging.WARNING)

# paddleocr must be imported before paddle: paddleocr pulls in torch, and on Windows torch
# fails to load its DLLs (WinError 127) if paddle was imported first
from paddleocr import TextDetection, TextRecognition

import cv2

from paddle_engine import boxes, masked, recognize
from tesseract_engine import image

sys.stdout.reconfigure(encoding="utf-8")

SYNTHETIC_DIR = Path(__file__).parent / "synthetic"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_CONFIDENCE = 0.6
DET_MODEL = "PP-OCRv6_medium_det"
AR_REC_MODEL = "arabic_PP-OCRv5_mobile_rec"
EN_REC_MODEL = "en_PP-OCRv5_mobile_rec"
# The PaddleOCR pipeline's detection settings. The module's own defaults shrink the image
# to 960 px on its long side and loosen boxes (unclip 2.0): small text is lost.
DET_PARAMS = {"limit_side_len": 64, "limit_type": "min", "thresh": 0.3, "box_thresh": 0.6, "unclip_ratio": 1.5}
# Padding around each line crop, x the line height: detector boxes are already loose.
CROP_PAD = 0.1


class PaddleEngine:
    """The detector and the two recognizers, loaded once."""

    def __init__(self, device, cpu_threads=None):
        # enable_mkldnn=False works around a oneDNN crash in paddlepaddle 3.3 on CPU
        common = {"device": device, "enable_mkldnn": False}
        if cpu_threads is not None:
            common["cpu_threads"] = cpu_threads
        self.det = TextDetection(model_name=DET_MODEL, **DET_PARAMS, **common)
        self.rec_ar = recognize.Recognizer(TextRecognition(model_name=AR_REC_MODEL, **common))
        self.rec_en = recognize.Recognizer(TextRecognition(model_name=EN_REC_MODEL, **common))


def extract_lines(ocr, image_path):
    """OCR lines with their boxes as [x0, y0, x1, y1] in the uploaded image's pixels."""
    img, (ox, oy, scale), blocks = image.prepare_with_blocks(image.load(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found = ocr.det.predict(img)
    line_boxes = [boxes.from_polygon(p) for p in (found[0]["dt_polys"] if found else [])]
    line_boxes = boxes.merge_overlapping([b for b in line_boxes if b[2] - b[0] > 3 and boxes.height(b) > 3])

    crops = [boxes.crop(img, b, int(CROP_PAD * boxes.height(b))) for b in line_boxes]
    lines = [{"box": b, "ar": ar, "en": en}
             for b, ar, en in zip(line_boxes, ocr.rec_ar.read(crops), ocr.rec_en.read(crops))]
    lines = masked.read_masked_lines(ocr.rec_ar, ocr.rec_en, img, gray, blocks, lines, MIN_CONFIDENCE)

    out = []
    for line in lines:
        text, conf = line.get("final") or recognize.choose(line["ar"], line["en"])
        if text and conf >= MIN_CONFIDENCE:
            x0, y0, x1, y1 = line["box"]
            box = [round(ox + x0 / scale), round(oy + y0 / scale), round(ox + x1 / scale), round(oy + y1 / scale)]
            out.append({"text": text, "score": conf, "box": box})
    out.sort(key=lambda l: (l["box"][1], l["box"][0]))
    return out


def extract_text(ocr, image_path):
    return "\n".join(line["text"] for line in extract_lines(ocr, image_path))


def detect_device():
    """'gpu:0' when paddlepaddle is a CUDA build and sees a GPU, otherwise 'cpu'."""
    import paddle  # only after paddleocr, see the import at the top

    try:
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu:0"
    except Exception:  # broken CUDA driver/runtime -> CPU still works
        pass
    return "cpu"


def create_ocr(cpu_threads=None, device=None):
    return PaddleEngine(device or detect_device(), cpu_threads)


def main():
    ocr = create_ocr()
    for image_path in sorted(p for p in SYNTHETIC_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        text = extract_text(ocr, image_path)
        image_path.with_suffix(".paddleocr.txt").write_text(text, encoding="utf-8")
        print(f"--- {image_path.name} ---\n{text}\n")


if __name__ == "__main__":
    main()
