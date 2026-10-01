"""Every line is read by two recognizers, and the more plausible reading is kept.

  - arabic_PP-OCRv5_mobile_rec: the only one that reads Arabic; reads Latin and digits too,
    but drops word breaks ("MahmoudFA", "Ling Expen")
  - en_PP-OCRv5_mobile_rec: far better on Latin text, e-mails and digits; turns Arabic
    into junk ("Jgaalll", "ス")

Reads can be limited to a character set (a CTC whitelist, like Tesseract's
tessedit_char_whitelist): the masked-name reader uses it for single initials, which the
Arabic model otherwise reads as Latin look-alikes ("م" -> "p").
"""
import logging
import re
from difflib import SequenceMatcher

import numpy as np

log = logging.getLogger(__name__)

ARABIC = re.compile("[ء-يٱ-ۓ]")
LATIN = re.compile(r"[A-Za-z]")
# anything outside ASCII and the Arabic block: what the Latin model makes of Arabic glyphs
JUNK = re.compile(r"[^\u0000-\u007f؀-ۿ]")
# OCR noise at the ends of a line (the edge of an icon, a table rule)
EDGE_NOISE = "-_.,:;'\"`|!()[]{}~ـ،؛ "
AR_LETTERS = "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"
LA_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LA_LETTERS = LA_UPPER + LA_UPPER.lower()
BATCH_SIZE = 16
# On a Latin-only line the Latin model wins unless it is this much less confident.
LATIN_MARGIN = 0.05
# A Latin part of a mixed line is taken from the Latin model when it is at least this
# similar to what the Arabic model read there.
MIXED_SIMILARITY = 0.6


class Whitelist:
    """Wraps a recognizer's CTC decoder: while ``allowed`` is set, every other character
    gets probability 0, so the best path only uses allowed characters."""

    def __init__(self, decoder):
        self.decoder = decoder
        self.chars = decoder.character  # index 0 is the CTC blank
        self.allowed = None
        self._masks = {}

    def mask(self, charset):
        if charset not in self._masks:
            self._masks[charset] = np.array([i == 0 or c in charset for i, c in enumerate(self.chars)],
                                            dtype=np.float32)
        return self._masks[charset]

    def __call__(self, pred, **kwargs):
        if self.allowed is not None:
            pred = [np.asarray(pred[0]) * self.allowed]
        return self.decoder(pred, **kwargs)


class Recognizer:
    """One PaddleOCR TextRecognition model, optionally limited to a character set."""

    def __init__(self, model):
        self.model = model
        predictor = getattr(model, "paddlex_predictor", None)
        if predictor is not None and hasattr(getattr(predictor, "post_op", None), "character"):
            self.whitelist = predictor.post_op = Whitelist(predictor.post_op)
        else:  # another paddleocr version: reads still work, only unrestricted
            log.warning("cannot restrict %s to a character set", type(model).__name__)
            self.whitelist = None

    def read(self, crops, charset=None):
        """[(text, confidence 0-1)] for each crop."""
        if not crops:
            return []
        if charset and self.whitelist:
            self.whitelist.allowed = self.whitelist.mask(charset)
        try:
            out = self.model.predict(crops, batch_size=BATCH_SIZE)
        finally:
            if self.whitelist:
                self.whitelist.allowed = None
        return [(str(r["rec_text"]), float(r["rec_score"])) for r in out]


def clean(text):
    return re.sub(r"\s+", " ", text).strip().strip(EDGE_NOISE)


def fix_text(text):
    # InstaPay handle: an 'o' among the trailing digits is a zero (narrow 0 in the app font)
    text = re.sub(r"(?<=\d)[oO](?=[\doO]*@)", "0", text)
    # 'عبد الرحمن' read for 'عبدالرحمن': the gap after the non-joining dal is no word break
    return re.sub(r"عبد\s+ال", "عبدال", text)


def choose(ar, en):
    """The line's text and confidence from its (text, conf) readings by both models."""
    text, conf = _choose(*ar, *en)
    return fix_text(text), conf


def _choose(a, ac, e, ec):
    e_raw = " ".join(e.split())
    a, e = clean(a), clean(e)
    if not e or JUNK.search(e):
        return a, ac
    if not a:
        return e, ec
    if ARABIC.search(a):  # only the Arabic model reads Arabic
        if LATIN.search(a) and (mixed := _mixed(a, e_raw)):
            return mixed, ac
        return a, ac
    # same letters, different word breaks: the models drop spaces, they don't invent them
    if a.replace(" ", "") == e.replace(" ", "") and a != e:
        return (a, ac) if a.count(" ") > e.count(" ") else (e, ec)
    if ec >= ac - LATIN_MARGIN:
        return e, ec
    return a, ac


def _mixed(a, e):
    """A line mixing Latin and Arabic: the Arabic model's reading with its Latin part taken
    from the Latin model, which reads only that part ('Ling Expen  عيدية' + 'Living
    Expenses - ' -> 'Living Expenses - عيدية'). None when the Latin reading doesn't match."""
    if m := re.fullmatch(r"([^؀-ۿ]*?)\s*([؀-ۿ][؀-ۿ\s]*)", a):
        latin, arabic, latin_first = m.group(1), m.group(2), True
    elif m := re.fullmatch(r"([؀-ۿ][؀-ۿ\s]*?)\s*([^؀-ۿ]*)", a):
        arabic, latin, latin_first = m.group(1), m.group(2), False
    else:
        return None
    if not LATIN.search(latin):
        return None

    def letters(s):
        return re.sub(r"[^a-z]", "", s.lower())

    # the run of Latin-model words, from the Latin side, that best matches the Latin part
    words = e.split() if latin_first else e.split()[::-1]
    best, n = 0.0, 0
    for k in range(1, len(words) + 1):
        part = " ".join(words[:k] if latin_first else words[:k][::-1])
        score = SequenceMatcher(None, letters(latin), letters(part)).ratio()
        if score > best:
            best, n = score, k
    # punctuation right after the Latin words ("Living Expenses - ") belongs to them
    while n < len(words) and not re.search(r"[A-Za-z0-9]", words[n]):
        n += 1
    if best < MIXED_SIMILARITY:
        return None
    part = " ".join(words[:n] if latin_first else words[:n][::-1])
    return f"{part} {arabic.strip()}" if latin_first else f"{arabic.strip()} {part}"
