import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
logging.disable(logging.WARNING)

from paddleocr import PaddleOCR

sys.stdout.reconfigure(encoding="utf-8")

SYNTHETIC_DIR = Path(__file__).parent / "synthetic"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_CONFIDENCE = 0.6


def extract_text(ocr, image_path):
    res = ocr.predict(str(image_path))[0]
    return "\n".join(
        text.strip()
        for text, score in zip(res["rec_texts"], res["rec_scores"])
        if score >= MIN_CONFIDENCE and text.strip()
    )


def create_ocr():
    # enable_mkldnn=False works around a oneDNN crash in paddlepaddle 3.3 on CPU
    return PaddleOCR(
        lang="ar",
        ocr_version="PP-OCRv5",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        enable_mkldnn=False,
    )


def main():
    ocr = create_ocr()
    for image_path in sorted(p for p in SYNTHETIC_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        text = extract_text(ocr, image_path)
        image_path.with_suffix(".paddleocr.txt").write_text(text, encoding="utf-8")
        print(f"--- {image_path.name} ---\n{text}\n")


if __name__ == "__main__":
    main()
