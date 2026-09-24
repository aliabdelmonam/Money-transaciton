import logging
import sys
from pathlib import Path

logging.disable(logging.WARNING)

import easyocr

sys.stdout.reconfigure(encoding="utf-8")

SYNTHETIC_DIR = Path(__file__).parent / "synthetic"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_CONFIDENCE = 0.6


def extract_text(reader, image_path):
    results = reader.readtext(str(image_path))
    return "\n".join(
        text.strip()
        for _, text, score in results
        if score >= MIN_CONFIDENCE and text.strip()
    )


def main():
    reader = easyocr.Reader(["ar", "en"], gpu=False, verbose=False)
    for image_path in sorted(p for p in SYNTHETIC_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        text = extract_text(reader, image_path)
        image_path.with_suffix(".easyocr.txt").write_text(text, encoding="utf-8")
        print(f"--- {image_path.name} ---\n{text}\n")


if __name__ == "__main__":
    main()
