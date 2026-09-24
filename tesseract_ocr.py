import sys
from itertools import groupby
from pathlib import Path

import pytesseract
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

SYNTHETIC_DIR = Path(__file__).parent / "synthetic"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_CONFIDENCE = 60  # Tesseract reports word confidence on a 0-100 scale


def extract_text(image_path):
    data = pytesseract.image_to_data(
        Image.open(image_path), lang="ara+eng", output_type=pytesseract.Output.DICT
    )
    words = [
        (key, text.strip())
        for key, text, conf in zip(
            zip(data["block_num"], data["par_num"], data["line_num"]),
            data["text"],
            data["conf"],
        )
        if float(conf) >= MIN_CONFIDENCE and text.strip()
    ]
    return "\n".join(" ".join(w for _, w in group) for _, group in groupby(words, key=lambda w: w[0]))


def main():
    for image_path in sorted(p for p in SYNTHETIC_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        text = extract_text(image_path)
        image_path.with_suffix(".tesseract.txt").write_text(text, encoding="utf-8")
        print(f"--- {image_path.name} ---\n{text}\n")


if __name__ == "__main__":
    main()
