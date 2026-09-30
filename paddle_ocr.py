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


def extract_lines(ocr, image_path):
    """OCR lines with their boxes as [x0, y0, x1, y1] in pixels."""
    res = ocr.predict(str(image_path))[0]
    return [
        {"text": text.strip(), "score": float(score), "box": [int(v) for v in box]}
        for text, score, box in zip(res["rec_texts"], res["rec_scores"], res["rec_boxes"])
        if score >= MIN_CONFIDENCE and text.strip()
    ]


def extract_text(ocr, image_path):
    return "\n".join(line["text"] for line in extract_lines(ocr, image_path))


def detect_device():
    """'gpu:0' when paddlepaddle is a CUDA build and sees a GPU, otherwise 'cpu'."""
    # paddle is imported here, after paddleocr: on Windows torch fails to load its DLLs
    # (WinError 127) if paddle was imported first
    import paddle

    try:
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu:0"
    except Exception:  # broken CUDA driver/runtime -> CPU still works
        pass
    return "cpu"


def create_ocr(cpu_threads=None, device=None):
    # enable_mkldnn=False works around a oneDNN crash in paddlepaddle 3.3 on CPU
    extra = {} if cpu_threads is None else {"cpu_threads": cpu_threads}
    return PaddleOCR(
        lang="ar",
        ocr_version="PP-OCRv5",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        enable_mkldnn=False,
        device=device or detect_device(),
        **extra,
    )


def main():
    ocr = create_ocr()
    for image_path in sorted(p for p in SYNTHETIC_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        text = extract_text(ocr, image_path)
        image_path.with_suffix(".paddleocr.txt").write_text(text, encoding="utf-8")
        print(f"--- {image_path.name} ---\n{text}\n")


if __name__ == "__main__":
    main()

