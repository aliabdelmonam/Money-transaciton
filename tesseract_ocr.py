"""Tesseract OCR: receipt lines with their boxes, the same interface as paddle_ocr.

  1. image.prepare: dark mode / photo lighting / margins evened out, text scaled to ~28 px
     tall, tilt removed
  2. segment.find_blocks: text lines found with OpenCV morphology, icons dropped by colour
  3. every line read on its own with the English and the Arabic model; the more plausible
     script wins
  4. lines that need more care re-read with their own strategy (reader.py): addresses,
     digit strings and dates voted across scales with a whitelist, masked names rebuilt
     around their asterisks, the headline amount split by colour, mixed-script lines
     read with the combined model

Tesseract runs in-process through libtesseract when it can (models loaded once), else
one tesseract.exe per read.
"""
import logging
import os
import re
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Lines are read in parallel threads: OpenMP threads inside each read would only fight
# them for the CPU. Must be set before libtesseract loads.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import cv2
import numpy as np
import pytesseract

from tesseract_engine import fields, image, segment, tessapi
from tesseract_engine.reader import ALNUM, LOWER_ALNUM, LineReader, asterisk_runs, crop, pick_script

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# tessdata_best: the most accurate models (float LSTM); the Windows installer only ships
# tessdata_fast. Missing models are downloaded here on first use.
TESSDATA_DIR = Path(__file__).parent / "models" / "tessdata_best"
TESSDATA_URL = "https://github.com/tesseract-ocr/tessdata_best/raw/main/{lang}.traineddata"
LANGS = ("eng", "ara")
SYNTHETIC_DIR = Path(__file__).parent / "synthetic"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Line confidence, 0-1 like paddle_ocr but not on its scale: a correct light-grey Arabic
# line can score 0.25, and the extractor validates every value against its field type.
MIN_CONFIDENCE = 0.2
# A colourful block is more often a logo or icon than colour text: it must read surely.
COLOUR_MIN_CONFIDENCE = 0.8
# The headline amount is the tallest line with a digit, at least this x the median line.
AMOUNT_HEIGHT = 1.5
# A line with an English number is the amount over a taller one whose digits only the
# Arabic model sees when it is at least this share of its height.
AMOUNT_HEIGHT_TOLERANCE = 0.8
# Share of a block's box covered by ink above which it is a solid bar, not a text line.
SOLID_FILL = 0.6
# The English model's confidence on garbage read from Arabic glyphs stays below this.
MIXED_MIN_CONF = 60
# A single-script reading at least this confident is the line, mixed or not.
SURE_CONF = 85
# Lines this x taller than the median may mix text sizes ("You transferred 9 EGP").
MIXED_SIZE_HEIGHT = 1.3
CHAR_WIDTH = 0.5  # a character's width, x the line height

log = logging.getLogger(__name__)


class TesseractOCR:
    """Tesseract settings, the loaded engines and the thread pool lines are read in."""

    def __init__(self, workers, tessdata_dir=TESSDATA_DIR):
        cmd = os.environ.get("TESSERACT_CMD")
        if not cmd and not shutil.which("tesseract") and os.path.exists(DEFAULT_CMD):
            cmd = DEFAULT_CMD
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        try:
            for lang in LANGS:
                ensure_model(lang, tessdata_dir)
            # tesseract splits its config on whitespace: forward slashes, no quotes
            datapath = tessdata_dir.resolve().as_posix()
        except OSError as error:  # offline, GitHub down, ...: still read, just less accurately
            log.warning("could not get tessdata_best models (%s) -> using the installed (fast, less accurate) ones",
                        error)
            datapath = None
        engines = tessapi.Engines.create(datapath, workers)
        if engines is None:
            log.warning("libtesseract not available -> one tesseract.exe per read (about 4x slower)")
        self.backend = "libtesseract" if engines else "cli"
        self.reader = LineReader(engines, datapath)
        self.pool = ThreadPoolExecutor(max_workers=workers)


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


def find_amount(blocks, gray):
    """The headline amount: the tallest line that reads as a number, clearly taller than
    body text. Never an address line, nor a solid bar or button. The Arabic model finds
    digits in any jumble (an icon a little taller than the amount), so a line of about the
    same height whose English reading has the number beats one whose digits only the
    Arabic model sees."""
    if not blocks:
        return None
    median_h = float(np.median([b.h for b in blocks]))
    ink = segment.ink_threshold(gray)

    def tallest(cands):
        best = max(cands, key=lambda b: b.h, default=None)
        return best if best is not None and best.h >= AMOUNT_HEIGHT * median_h else None

    def english(b):
        # real digits: "I °c oo" is no number although I and o are digit look-alikes
        n = sum(ch.isdigit() for ch in b.extra["eng"].text)
        return n >= 2 or (n == 1 and bool(fields.amount(b.extra["eng"].text)))

    usable = [b for b in blocks
              if (gray[b.y:b.y2, b.x:b.x2] < ink).mean() < SOLID_FILL  # text covers well under half its box
              and not any("@" in b.extra[k].text for k in ("eng", "ara"))]
    best = tallest([b for b in usable if any(ch.isdigit() for ch in b.extra["eng"].text + b.extra["ara"].text)])
    if best is None or english(best):
        return best
    return tallest([b for b in usable if english(b) and b.h >= AMOUNT_HEIGHT_TOLERANCE * best.h]) or best


def is_mixed(eng, ara):
    """Worth a combined eng+ara read: each model read words of its own script, the English
    one surely, and neither read the whole line surely on its own."""
    return (MIXED_MIN_CONF <= eng.conf < SURE_CONF and ara.conf < SURE_CONF
            and bool(re.search(r"[A-Za-z]{3,}", eng.text) and re.search("[\u0600-\u06FF]{2,}", ara.text)))


def best_reading(reader, img, gray, b, is_amount, median_h):
    """Best reading of one block, by what its first-pass readings look like."""
    eng, ara = b.extra["eng"], b.extra["ara"]
    if is_amount and (r := reader.read_amount(img, gray, b)):
        return r
    if "@" in eng.text and fields.email(eng.text):
        r = reader.read_voted(gray, b, "eng", LOWER_ALNUM + "._-@", fields.email)
        if r:
            return r
    if runs := asterisk_runs(gray, b):
        return reader.read_masked_name(gray, b, runs, eng, ara)
    if fields.digits(eng.text, 6):
        r = reader.read_voted(gray, b, "eng", "0123456789", lambda t: fields.digits(t, 6))
        if r:
            return r
    if fields.date(eng.text) or fields.date(ara.text):
        r = reader.read_voted(gray, b, "eng", None, fields.date)
        if r:
            return r
    if fields.code(eng.text):
        r = reader.read_voted(gray, b, "eng", ALNUM, fields.code)
        if r:
            return r
    if is_mixed(eng, ara):
        return reader.read_mixed(gray, b, eng, ara)
    best, lang = pick_script(eng, ara, b.w / (CHAR_WIDTH * b.h))
    if lang == "eng" and b.h >= MIXED_SIZE_HEIGHT * median_h and len(best.text.split()) >= 3:
        best = reader.reread_small_words(gray, b, best)
    return best


def extract_lines(ocr, image_path):
    """OCR lines with their boxes as [x0, y0, x1, y1] in the uploaded image's pixels."""
    img, (ox, oy, scale) = image.prepare(image.load(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blocks = segment.find_blocks(img)
    reader = ocr.reader

    # first pass: every line with both models; the re-reads decide by these readings
    for b, (eng, ara) in zip(blocks, ocr.pool.map(lambda b: reader.read_both(crop(gray, b)), blocks)):
        b.extra["eng"], b.extra["ara"] = eng, ara
    amount = find_amount(blocks, gray)
    median_h = float(np.median([b.h for b in blocks])) if blocks else 0.0
    readings = ocr.pool.map(lambda b: best_reading(reader, img, gray, b, b is amount, median_h), blocks)

    kept = []
    for b, r in zip(blocks, readings):
        # colour blocks are more often logos than text; the amount was verified by its split
        floor = COLOUR_MIN_CONFIDENCE if b.extra.get("colour") and b is not amount else MIN_CONFIDENCE
        if r.text and r.conf / 100 >= floor:
            kept.append((b, r))
    # a kept colour block already holds the neutral text found inside it
    colour = [b for b, _ in kept if b.extra.get("colour")]
    kept = [(b, r) for b, r in kept if b.extra.get("colour") or not any(inside(b, c) for c in colour)]

    out = []
    for b, r in kept:
        box = [round(ox + b.x / scale), round(oy + b.y / scale), round(ox + b.x2 / scale), round(oy + b.y2 / scale)]
        out.append({"text": r.text, "score": r.conf / 100, "box": box})
    return out


def inside(a, b):
    """Most of block ``a`` lies within block ``b``."""
    w = max(0, min(a.x2, b.x2) - max(a.x, b.x))
    h = max(0, min(a.y2, b.y2) - max(a.y, b.y))
    return w * h >= 0.8 * a.w * a.h


def extract_text(ocr, image_path):
    return "\n".join(line["text"] for line in extract_lines(ocr, image_path))


def detect_device():
    """Tesseract only runs on the CPU."""
    return "cpu"


def create_ocr(cpu_threads=None, device=None):
    if device not in (None, "cpu"):
        raise ValueError(f"Tesseract only runs on the CPU, not {device!r}")
    return TesseractOCR(workers=cpu_threads or os.cpu_count() or 4)


def main():
    ocr = create_ocr()
    for image_path in sorted(p for p in SYNTHETIC_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        text = extract_text(ocr, image_path)
        image_path.with_suffix(".tesseract.txt").write_text(text, encoding="utf-8")
        print(f"--- {image_path.name} ---\n{text}\n")


if __name__ == "__main__":
    main()
