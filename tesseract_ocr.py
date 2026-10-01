import logging
import os
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytesseract

sys.stdout.reconfigure(encoding="utf-8")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# tessdata_best: the most accurate models (float LSTM); the Windows installer only ships tessdata_fast.
# Missing models are downloaded here on first use.
TESSDATA_DIR = Path(__file__).parent / "models" / "tessdata_best"
TESSDATA_URL = "https://github.com/tesseract-ocr/tessdata_best/raw/main/{lang}.traineddata"
SYNTHETIC_DIR = Path(__file__).parent / "synthetic"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_CONFIDENCE = 0.6  # mean word confidence of a line, 0-1 like paddle_ocr
LANG = "ara+eng"
PSM = 11              # sparse text: receipts are scattered label/value pairs
SMALL_WORD = 0.75     # a word shorter than this * the tallest word on its line is re-read alone
WORD_THREADS = 4      # small words re-read at once
MIN_LONG_SIDE = 1600  # smaller images are upscaled to this
GAMMA = 3             # > 1 darkens grey text; 3 read the most fields on synthetic/


log = logging.getLogger(__name__)


class TesseractOCR:
    """Tesseract settings; the engine itself is a tesseract.exe call per image."""

    def __init__(self, lang=LANG, psm=PSM, tessdata_dir=TESSDATA_DIR):
        self.lang = lang
        self.config = f"--oem 1 --psm {psm}"
        self.word_config = "--oem 1 --psm 8"  # psm 8: a single word
        try:
            for name in lang.split("+"):
                ensure_model(name, tessdata_dir)
        except OSError as error:  # offline, GitHub down, ...: still read, just less accurately
            log.warning("could not get tessdata_best models (%s) -> using the installed (fast, less accurate) ones",
                        error)
        else:
            self.config += f" --tessdata-dir {tessdata_dir.as_posix()}"
            self.word_config += f" --tessdata-dir {tessdata_dir.as_posix()}"


def ensure_model(lang, tessdata_dir=TESSDATA_DIR):
    """Path to ``<lang>.traineddata`` in ``tessdata_dir``, downloaded from tessdata_best if missing."""
    path = tessdata_dir / f"{lang}.traineddata"
    if path.is_file():
        return path
    tessdata_dir.mkdir(parents=True, exist_ok=True)
    url = TESSDATA_URL.format(lang=lang)
    log.info("downloading %s -> %s", url, path)
    # several OCR workers may start at once: each writes its own temp file, the rename is atomic
    tmp = path.with_name(f"{path.name}.{os.getpid()}.part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def prepare(image):
    """Grey, dark text on a light background, upscaled if small: what Tesseract reads best."""
    scale = MIN_LONG_SIDE / max(image.shape[:2])
    if scale > 1:  # small screenshots: Tesseract wants ~30 px tall text
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    grey = image.min(axis=2)  # coloured text (orange amounts) becomes dark too
    grey = cv2.normalize(grey, None, 0, 255, cv2.NORM_MINMAX)
    if np.mean(grey) < 127:  # dark mode screenshot
        grey = 255 - grey
    # darken mid greys: light grey labels ("Reference", "From") are otherwise binarised away
    return (255 * (grey / 255.0) ** GAMMA).astype(np.uint8)


def read_word(image, box, lang, config):
    """Re-read one word on its own; (text, confidence 0-100), ('', -1) if nothing was read."""
    x0, y0, x1, y1 = box
    pad = max(4, (y1 - y0) // 3)
    crop = image[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    data = pytesseract.image_to_data(crop, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    found = [(t.strip(), float(c)) for t, c in zip(data["text"], data["conf"]) if t.strip() and float(c) >= 0]
    if not found:
        return "", -1.0
    return " ".join(t for t, _ in found), sum(c for _, c in found) / len(found)


def extract_lines(ocr, image_path):
    """OCR lines with their boxes as [x0, y0, x1, y1] in pixels."""
    original = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    image = prepare(original)
    scale = image.shape[0] / original.shape[0]
    data = pytesseract.image_to_data(image, lang=ocr.lang, config=ocr.config, output_type=pytesseract.Output.DICT)
    words = [
        {"text": t.strip(), "conf": float(c), "box": [x, y, x + w, y + h], "line": key}
        for t, c, x, y, w, h, *key in zip(data["text"], data["conf"], data["left"], data["top"], data["width"],
                                          data["height"], data["block_num"], data["par_num"], data["line_num"])
        if t.strip() and float(c) >= 0
    ]
    lines = {}
    for word in words:
        lines.setdefault(tuple(word["line"]), []).append(word)

    # Tesseract sizes a line by its tallest glyphs, so a smaller word next to big ones is misread:
    # the "EGP" after a big "6,000" comes out as Arabic ("مع"). Read those words on their own.
    small = [
        word for line in lines.values() if len(line) > 1
        for word in line
        if word["box"][3] - word["box"][1] < SMALL_WORD * max(w["box"][3] - w["box"][1] for w in line)
    ]
    # every read is its own tesseract.exe, so they run in parallel
    with ThreadPoolExecutor(WORD_THREADS) as pool:
        rereads = list(pool.map(lambda w: read_word(image, w["box"], ocr.lang, ocr.word_config), small))
    for word, (text, conf) in zip(small, rereads):
        if conf > word["conf"] and " " not in text:  # one word in, one word out: else it read noise
            word["text"], word["conf"] = text, conf

    out = []
    for line in lines.values():
        score = sum(w["conf"] for w in line) / len(line) / 100
        if score < MIN_CONFIDENCE:
            continue
        box = [min(w["box"][0] for w in line), min(w["box"][1] for w in line),
               max(w["box"][2] for w in line), max(w["box"][3] for w in line)]
        box = [round(v / scale) for v in box]  # back to the original image's pixels
        out.append({"text": " ".join(w["text"] for w in line), "score": score, "box": box})
    return out


def extract_text(ocr, image_path):
    return "\n".join(line["text"] for line in extract_lines(ocr, image_path))


def detect_device():
    """Tesseract only runs on the CPU."""
    return "cpu"


def create_ocr(cpu_threads=None, device=None):
    if device not in (None, "cpu"):
        raise ValueError(f"Tesseract only runs on the CPU, not {device!r}")
    if cpu_threads is not None:
        # read by tesseract.exe (OpenMP), which inherits this process's environment
        os.environ["OMP_THREAD_LIMIT"] = str(cpu_threads)
    return TesseractOCR()


def main():
    ocr = create_ocr()
    for image_path in sorted(p for p in SYNTHETIC_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        text = extract_text(ocr, image_path)
        image_path.with_suffix(".tesseract.txt").write_text(text, encoding="utf-8")
        print(f"--- {image_path.name} ---\n{text}\n")


if __name__ == "__main__":
    main()
