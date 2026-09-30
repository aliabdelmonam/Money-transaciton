#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
receipt_ocr.py - Egyptian InstaPay / e-wallet receipt reader (Arabic + English)

Pipeline
  1. PaddleOCR detector (PP-OCRv5 server det) finds the text lines.
  2. Every line is read by the PaddleOCR Arabic model (Arabic + English) AND the
     PP-OCRv6 English model. The best reading becomes the token text, all readings
     are kept in `alts` so labels / anchors can match on any of them.
     (old engine: EasyOCR CRAFT detector + Tesseract/EasyOCR readers - commented out below)
  3. Tokens are grouped into visual lines using their bounding boxes.
  4. Fields: sentence patterns -> from/to anchors (sender/receiver blocks)
     -> "label : value" on the same line (or the line above/below).
  5. amount / fees / total are cross-checked; missing ones derived.

Edit the CONFIG block and run:   python receipt_ocr.py
"""

# =============================== CONFIG =====================================
INPUT_DIR = r"./synthetic"             # folder with receipt images
OUTPUT_DIR = r"./output"            # results go here
# TESSERACT_CMD = None                # e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe" (None = on PATH)
# TESSDATA_DIR = None                 # folder containing ara.traineddata + eng.traineddata (tessdata_best)
USE_GPU = True                     # True if you have CUDA + paddlepaddle-gpu build (falls back to CPU)

# PaddleOCR models (downloaded to ~/.paddlex/official_models on first run)
PADDLE_DET_MODEL = "PP-OCRv5_server_det"            # text line detector
PADDLE_AR_REC_MODEL = "arabic_PP-OCRv5_mobile_rec"  # Arabic + English + digits
PADDLE_EN_REC_MODEL = "PP-OCRv6_medium_rec"         # English / digits (None = Arabic model only)
PADDLE_MIN_CONF = 0.30              # drop readings below this (0-1)
PADDLE_EN_MARGIN = 0.05             # latin-only line: English model wins unless this much less confident
PADDLE_CROP_PAD = 0.15              # padding around each line crop (ratio of box height)
PADDLE_BATCH_SIZE = 16
PADDLE_MERGE_BOXES = False          # Paddle already returns whole lines; True = also merge close boxes

TARGET_LONG_SIDE = 1600             # small screenshots are upscaled to this
MAX_LONG_SIDE = 2600                # huge photos are downscaled to this
# TESS_MIN_CONF = 70                  # Tesseract reading accepted above this (0-100)
# EASY_MIN_CONF = 0.30                # EasyOCR reading used as override above this (0-1)
MERGE_GAP_RATIO = 1.5               # merge boxes on a line if gap <= ratio * box height
LINE_Y_TOLERANCE = 0.6              # same line if |dy| <= tol * height
FUZZY_LABEL_THRESHOLD = 82          # fuzzy match score for labels (OCR typos)
MAX_PARTY_LINES = 4                 # max lines in a sender/receiver block
PARTY_GAP_RATIO = 3.5               # a vertical gap bigger than this * height ends a block
SAVE_DEBUG_IMAGES = True            # boxes coloured by role -> output/debug
SAVE_TEXT_FILES = True              # raw OCR text per image -> output/text
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
# ============================================================================

import os
import re
import csv
import json
import glob
import difflib
import warnings
import unicodedata
from dataclasses import dataclass
from datetime import datetime

warnings.filterwarnings("ignore")

import cv2
import numpy as np
# import easyocr                      # old engine (EasyOCR + Tesseract), replaced by PaddleOCR
# import pytesseract
# from pytesseract import Output

try:
    from rapidfuzz import fuzz as _fuzz

    def ratio(a, b):
        return float(_fuzz.ratio(a, b))
except ImportError:
    def ratio(a, b):
        return difflib.SequenceMatcher(None, a, b).ratio() * 100.0


# ============================ PROVIDER KNOWLEDGE ============================
# field -> label variants (Arabic + English). Add new wording here.
GENERIC_LABELS = {
    "amount": ["amount", "transfer amount", "transferred amount", "amount sent",
               "المبلغ", "المبلغ المحول", "المبلغ الإجمالي المحول", "قيمة التحويل", "قيمة المعاملة"],
    "fees": ["fees", "fee", "transaction fees", "service fees",
             "رسوم", "الرسوم", "رسوم المعاملة", "رسوم التحويل", "رسوم الخدمة"],
    "total": ["total", "total amount", "total amount deducted", "total deducted",
              "المبلغ الكلي", "الإجمالي", "المبلغ الإجمالي", "إجمالي المبلغ", "إجمالي المبلغ المخصوم"],
    "reference": ["reference", "reference number", "reference no", "ref", "ref no",
                  "transaction id", "transaction number", "transaction ref",
                  "المرجع", "الرقم المرجعي", "رقم المرجع", "رقم العملية", "رقم المعاملة"],
    "date": ["date", "date time", "transaction time", "transaction date", "time",
             "التاريخ", "تاريخ العملية", "التاريخ والوقت", "الوقت"],
    "note": ["note", "notes", "description", "purpose", "ملاحظة", "ملاحظات", "الغرض"],
    "sender.name": ["sender name", "from name", "اسم المرسل", "اسم المحول"],
    "sender.phone": ["sender number", "sender mobile", "wallet number", "رقم المرسل", "رقم المحفظة"],
    "receiver.name": ["receiver name", "beneficiary name", "recipient name",
                      "اسم المستقبل", "اسم المستفيد", "اسم المرسل إليه"],
    "receiver.phone": ["receiver number", "receiver mobile", "recipient number", "beneficiary number",
                       "رقم المستقبل", "رقم المستفيد", "رقم المرسل إليه"],
}

# detect: keywords that identify the provider (any language, matched on normalized text)
# label_overrides: label -> field, for wording that means something different in this app
# sentences: extra regexes with named groups amount / sender_phone / receiver_phone
PROVIDERS = {
    "instapay": {"detect": ["@instapay", "instapay", "انستاباي", "powered by", "ipn"],
                 "label_overrides": {}, "sentences": []},
    "axis": {"detect": ["axis", "الدفع تم بنجاح"],
             "label_overrides": {"رقم المحفظة": "sender.phone", "رقم المرسل إليه": "receiver.phone"},
             "sentences": []},
    # guessed name - screenshot with "Total Amount Deducted"
    "vodafone_cash": {"detect": ["total amount deducted", "payment has been processed"],
                      "label_overrides": {"total amount": "amount"}, "sentences": []},
    "wallet_simple": {"detect": ["you transferred", "you have transferred"],
                      "label_overrides": {}, "sentences": []},
}

_NUM = r"(?P<amount>\d[\d,]*(?:\.\d+)?)"
_PH_R = r"(?P<receiver_phone>\+?\d[\d ]{8,14}\d)"
_PH_S = r"(?P<sender_phone>\+?\d[\d ]{8,14}\d)"
GENERIC_SENTENCES = [
    r"you\s+(?:have\s+)?(?:successfully\s+)?(?:transferred|sent|paid)\s+(?:egp\s*)?" + _NUM
    + r"\s*(?:egp|le)?\.?\s+to\s+" + _PH_R,
    r"you\s+(?:have\s+)?received\s+(?:egp\s*)?" + _NUM + r"\s*(?:egp|le)?\.?\s+from\s+" + _PH_S,
    r"تم\s+(?:تحويل|ارسال)\s+(?:مبلغ\s+)?" + _NUM + r".{0,25}?(?:الي|الى)\s*" + _PH_R,
    r"تم\s+استلام\s+(?:مبلغ\s+)?" + _NUM + r".{0,25}?من\s*" + _PH_S,
]

ANCHOR_WORDS = {"sender": {"from", "من"}, "receiver": {"to", "الي"}}   # normalized forms

BANKS = {"banque misr": "Banque Misr", "بنك مصر": "Banque Misr",
         "national bank of egypt": "National Bank of Egypt", "nbe": "National Bank of Egypt",
         "البنك الاهلي": "National Bank of Egypt", "cib": "CIB", "البنك التجاري الدولي": "CIB",
         "qnb": "QNB", "banque du caire": "Banque du Caire", "بنك القاهره": "Banque du Caire",
         "alexbank": "AlexBank", "aaib": "AAIB", "hsbc": "HSBC", "adib": "ADIB",
         "telda": "Telda", "fawry": "Fawry"}

MONEY_FIELDS = ("amount", "fees", "total")
EMPTY_PARTY = ("name", "name_alt", "phone", "email", "bank", "account_type")
SUCCESS_WORDS = ["success", "بنجاح", "تمت العمليه", "تم الدفع"]
FAIL_WORDS = ["unsuccessful", "failed", "declined", "فشل", "مرفوض", "لم تتم"]


# ============================== TEXT HELPERS ================================
_BIDI = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_DIAC = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_AR_LET = re.compile(r"[\u0621-\u063A\u0641-\u064A\u0671-\u06D3]")
_LA_LET = re.compile(r"[A-Za-z]")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬", "01234567890123456789.,")


def clean_text(s):
    s = unicodedata.normalize("NFKC", _BIDI.sub("", str(s or "")))
    s = re.sub(r"\s+", " ", s)
    return s.strip(" -_|،;:")


def to_ascii_digits(s):
    return (s or "").translate(_DIGITS)


def _ar_norm(s):
    s = _DIAC.sub("", s)
    s = re.sub("[إأآٱ]", "ا", s)
    return s.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")


def norm(s):
    """aggressive normalization for matching (no punctuation)."""
    s = _ar_norm(to_ascii_digits(clean_text(s)).lower())
    s = re.sub(r"[^\w@*]+", " ", s)
    return s.strip()


def norm_soft(s):
    """normalization that keeps punctuation (numbers, sentences)."""
    return _ar_norm(to_ascii_digits(clean_text(s)).lower())


def script_counts(s):
    return len(_AR_LET.findall(s or "")), len(_LA_LET.findall(s or ""))


# ------------------------------ value parsers -------------------------------
_MONEY_RE = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?(?!\d)")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?20|0020)?0?(1[0125]\d{8})(?!\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+\s?@\s?[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*")


def find_phone(texts):
    for t in texts:
        s = to_ascii_digits(clean_text(t))
        for cand in (s, re.sub(r"[\s\-().]", "", s)):
            m = PHONE_RE.search(cand)
            if m:
                return "0" + m.group(1)
    return None


def find_email(texts):
    for t in texts:
        m = EMAIL_RE.search(clean_text(t))
        if m:
            return re.sub(r"\s", "", m.group(0)).lower()
    return None


def find_bank(texts):
    for t in texts:
        n = " " + norm(t) + " "
        for key, name in BANKS.items():
            if " " + norm(key) + " " in n:
                return name
    return None


def parse_money(texts):
    for t in texts:
        s = to_ascii_digits(clean_text(t))
        if not s:
            continue
        words = norm(s).split()
        if any(w in words for w in ("free", "مجانا", "مجاني")):
            return 0.0
        if find_phone([s]):
            continue
        # "EGP501" -> "EGP 501"
        s = re.sub(r"(?<=[A-Za-z\u0600-\u06FF])(?=\d)|(?<=\d)(?=[A-Za-z\u0600-\u06FF])", " ", s)
        best = None
        for m in _MONEY_RE.finditer(s):
            whole = m.group(1).replace(",", "")
            val = float(whole + ("." + m.group(2) if m.group(2) else ""))
            if best is None or len(whole) > best[0]:
                best = (len(whole), val)
        if best:
            return best[1]
    return None


def parse_reference(texts):
    for t in texts:
        s = to_ascii_digits(clean_text(t))
        runs = [r.strip("-") for r in re.findall(r"[A-Za-z0-9\-]{4,}", s)]
        runs = [r for r in runs if len(r) >= 4]
        if runs:
            return max(runs, key=len)
    return None


_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
           "sep": 9, "oct": 10, "nov": 11, "dec": 12,
           "يناير": 1, "فبراير": 2, "مارس": 3, "ابريل": 4, "مايو": 5, "يونيو": 6, "يوليو": 7,
           "اغسطس": 8, "سبتمبر": 9, "اكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12}
_OCR_FIX = str.maketrans({"v": "y", "0": "o", "1": "l", "5": "s", "8": "b"})


def month_num(w):
    w = norm(w)
    if not w:
        return None
    if w in _MONTHS:
        return _MONTHS[w]
    if w[:3] in _MONTHS:
        return _MONTHS[w[:3]]
    w2 = w.translate(_OCR_FIX)          # 'Mav' -> 'may'
    if w2[:3] in _MONTHS:
        return _MONTHS[w2[:3]]
    m = difflib.get_close_matches(w, list(_MONTHS), n=1, cutoff=0.75)
    return _MONTHS[m[0]] if m else None


def parse_datetime(text):
    """returns ISO string ('2026-09-19T13:37') or None. Date part required."""
    s = norm_soft(text)
    y = mo = d = None
    span = None
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?[\s\-/.,]*([a-z\u0621-\u064a]{3,9})\.?[\s\-/.,]*(\d{4})", s)
    if m and month_num(m.group(2)):
        d, mo, y, span = int(m.group(1)), month_num(m.group(2)), int(m.group(3)), m.span()
    if y is None:
        m = re.search(r"([a-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})", s)
        if m and month_num(m.group(1)):
            mo, d, y, span = month_num(m.group(1)), int(m.group(2)), int(m.group(3)), m.span()
    if y is None:
        m = re.search(r"(\d{4})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{1,2})", s)
        if m:
            y, mo, d = map(int, m.groups())
            span = m.span()
    if y is None:
        m = re.search(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})", s)
        if m:
            d, mo, y = map(int, m.groups())
            span = m.span()
            if y < 100:
                y += 2000
            if mo > 12 and d <= 12:
                d, mo = mo, d
    if y is None or not (1 <= mo <= 12 and 1 <= d <= 31 and 1990 <= y <= 2100):
        return None
    rest = s[:span[0]] + " " + s[span[1]:]
    hh = mi = 0
    has_time = False
    m = re.search(r"(\d{1,2})\s*[:.]\s*(\d{2})(?:\s*[:.]\s*\d{2})?\s*(a\.?\s?m|p\.?\s?m|p[1l|i]|ص|م)?", rest)
    if m:
        hh, mi = int(m.group(1)), int(m.group(2))
        ampm = (m.group(3) or "").replace(".", "").replace(" ", "")
        if (ampm.startswith("p") or ampm == "م") and hh < 12:
            hh += 12
        elif (ampm.startswith("a") or ampm == "ص") and hh == 12:
            hh = 0
        has_time = 0 <= hh <= 23 and 0 <= mi <= 59
        if not has_time:
            hh = mi = 0
    try:
        dt = datetime(y, mo, d, hh, mi)
    except ValueError:
        return None
    return dt.isoformat(timespec="minutes") if has_time else dt.date().isoformat()


NOT_NAMES = {norm(x) for x in ["powered by", "egp", "instapay", "انستاباي", "mobile wallet", "wallet",
                               "محفظة", "محفظة موبايل", "success", "successful", "transaction successful",
                               "بنجاح", "bank", "ج م", "le"]}


def clean_name(s):
    s = clean_text(s)
    s = re.sub(r"[^\w\s*.\-']", " ", s)
    s = re.sub(r"[\d_]", " ", s)
    return re.sub(r"\s+", " ", s).strip(" .-'")


def is_name(s):
    if not s or norm(s) in NOT_NAMES:
        return False
    letters = _AR_LET.findall(s) + _LA_LET.findall(s)
    if len(letters) < 3:
        return False
    return len({c.lower() for c in letters}) > 1       # rejects 'nnn' logo garbage


# ============================== DATA CLASSES ================================
@dataclass(eq=False)
class Token:
    text: str
    alts: list
    conf: float
    engine: str
    box: list          # x1, y1, x2, y2 (working image coords)
    role: str = ""
    line: int = -1

    @property
    def x1(self): return self.box[0]
    @property
    def y1(self): return self.box[1]
    @property
    def x2(self): return self.box[2]
    @property
    def y2(self): return self.box[3]
    @property
    def cx(self): return (self.box[0] + self.box[2]) / 2
    @property
    def cy(self): return (self.box[1] + self.box[3]) / 2
    @property
    def h(self): return max(1, self.box[3] - self.box[1])


def join_tokens(toks):
    if not toks:
        return ""
    ar, la = script_counts(" ".join(t.text for t in toks))
    order = sorted(toks, key=lambda t: t.x1, reverse=ar > la)   # RTL for Arabic lines
    return " ".join(t.text for t in order).strip()


class Line:
    def __init__(self, idx, tokens):
        self.idx = idx
        self.tokens = sorted(tokens, key=lambda t: t.x1)

    @property
    def cy(self): return float(np.mean([t.cy for t in self.tokens]))
    @property
    def h(self): return float(np.median([t.h for t in self.tokens]))

    def text(self, toks=None):
        return join_tokens(self.tokens if toks is None else toks)


def group_lines(tokens):
    rows = []
    for t in sorted(tokens, key=lambda t: t.cy):
        for row in reversed(rows[-3:]):
            rcy = np.mean([x.cy for x in row])
            rh = np.median([x.h for x in row])
            if abs(t.cy - rcy) <= LINE_Y_TOLERANCE * min(t.h, rh):
                row.append(t)
                break
        else:
            rows.append([t])
    lines = [Line(i, r) for i, r in enumerate(rows)]
    for ln in lines:
        for t in ln.tokens:
            t.line = ln.idx
    return lines


# ================================ OCR ENGINE ================================
def _should_merge(a, b):
    ha, hb = a[3] - a[1], b[3] - b[1]
    if min(ha, hb) <= 0:
        return False
    ov = min(a[3], b[3]) - max(a[1], b[1])
    if ov <= 0 or ov / min(ha, hb) < 0.6 or min(ha, hb) / max(ha, hb) < 0.45:
        return False
    gap = max(a[0], b[0]) - min(a[2], b[2])
    return gap <= MERGE_GAP_RATIO * max(ha, hb)


def merge_boxes(boxes):
    changed = True
    while changed:
        changed = False
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        out = []
        while boxes:
            a = boxes.pop(0)
            i = 0
            while i < len(boxes):
                if _should_merge(a, boxes[i]):
                    b = boxes.pop(i)
                    a = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    changed = True
                else:
                    i += 1
            out.append(a)
        boxes = out
    return boxes


# def choose_reading(e_txt, e_conf, t_txt, t_conf):
#     e_txt, t_txt = clean_text(e_txt), clean_text(t_txt)
#     if not t_txt:
#         return e_txt, e_conf, "easyocr"
#     if not e_txt:
#         return t_txt, t_conf / 100, "tesseract"
#     e_ar, e_la = script_counts(e_txt)
#     t_ar, t_la = script_counts(t_txt)
#     # Tesseract sometimes puts Latin junk into Arabic lines (and vice versa)
#     mismatch = (e_ar and not e_la and t_la) or (e_la and not e_ar and t_ar)
#     if mismatch and e_conf >= EASY_MIN_CONF:
#         return e_txt, e_conf, "easyocr"
#     if t_conf >= TESS_MIN_CONF:
#         return t_txt, t_conf / 100, "tesseract"
#     if e_conf >= 0.5:
#         return e_txt, e_conf, "easyocr"
#     return (t_txt, t_conf / 100, "tesseract") if t_conf / 100 >= e_conf else (e_txt, e_conf, "easyocr")
#
#
# class HybridOCR:
#     def __init__(self):
#         print("Loading EasyOCR (ar+en)...")
#         self.reader = easyocr.Reader(["ar", "en"], gpu=USE_GPU, verbose=False)
#         if TESSERACT_CMD:
#             pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
#         if TESSDATA_DIR:
#             os.environ["TESSDATA_PREFIX"] = os.path.abspath(TESSDATA_DIR)
#         self.tess_cfg = "--oem 1 --psm 7"
#         try:
#             available = set(pytesseract.get_languages(config=""))
#         except Exception as e:
#             print(f"[warn] Tesseract not usable ({e}) -> EasyOCR only (lower accuracy)")
#             available = set()
#         self.langs = available & {"ara", "eng"}
#         if available and "ara" not in available:
#             print("[warn] ara.traineddata not found -> Arabic accuracy will drop. Set TESSDATA_DIR.")
#         self.full_lang = "+".join(l for l in ("ara", "eng") if l in self.langs)
#
#     def detect(self, img):
#         hl, fl = self.reader.detect(img, width_ths=0.9, add_margin=0.08)
#         H, W = img.shape[:2]
#         boxes = []
#         for x1, x2, y1, y2 in (hl[0] if hl else []):
#             boxes.append([int(x1), int(y1), int(x2), int(y2)])
#         for poly in (fl[0] if fl else []):
#             xs = [p[0] for p in poly]
#             ys = [p[1] for p in poly]
#             boxes.append([int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))])
#         boxes = [[max(0, a), max(0, b), min(W, c), min(H, d)] for a, b, c, d in boxes]
#         boxes = [b for b in boxes if b[2] - b[0] > 4 and b[3] - b[1] > 4]
#         return merge_boxes(boxes)
#
#     def _easy(self, img, box):
#         x1, y1, x2, y2 = box
#         try:
#             r = self.reader.recognize(img, horizontal_list=[[x1, x2, y1, y2]], free_list=[], detail=1)
#         except Exception:
#             return "", 0.0
#         if not r:
#             return "", 0.0
#         return " ".join(x[1] for x in r), float(np.mean([x[2] for x in r]))
#
#     @staticmethod
#     def _prep_crop(img, box):
#         x1, y1, x2, y2 = box
#         H, W = img.shape[:2]
#         crop = img[max(0, y1 - 3):min(H, y2 + 3), max(0, x1 - 3):min(W, x2 + 3)]
#         g = crop.min(axis=2)                          # coloured text (orange EGP) becomes dark
#         g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
#         if np.mean(g) < 127:                          # white text on coloured button
#             g = 255 - g
#         g = cv2.copyMakeBorder(g, 12, 12, 12, 12, cv2.BORDER_REPLICATE)
#         if g.shape[0] < 64:
#             s = 64 / g.shape[0]
#             g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
#         return g
#
#     def _tess(self, g, lang):
#         if not lang:
#             return "", 0.0
#         try:
#             d = pytesseract.image_to_data(g, lang=lang, config=self.tess_cfg, output_type=Output.DICT)
#         except Exception:
#             return "", 0.0
#         words = [(str(t), float(c)) for t, c in zip(d["text"], d["conf"]) if str(t).strip() and float(c) >= 0]
#         if not words:
#             return "", 0.0
#         return " ".join(w for w, _ in words), sum(c for _, c in words) / len(words)
#
#     def _pick_lang(self, easy_text):
#         ar, la = script_counts(easy_text)
#         if ar and not la and "ara" in self.langs:
#             return "ara"
#         if la and not ar and "eng" in self.langs:
#             return "eng"
#         return self.full_lang
#
#     def read(self, img):
#         tokens = []
#         for box in self.detect(img):
#             e_txt, e_conf = self._easy(img, box)
#             t_txt, t_conf = "", 0.0
#             if self.langs:
#                 crop = self._prep_crop(img, box)
#                 lang = self._pick_lang(e_txt)
#                 t_txt, t_conf = self._tess(crop, lang)
#                 if t_conf < TESS_MIN_CONF and lang != self.full_lang:
#                     t2, c2 = self._tess(crop, self.full_lang)
#                     if c2 > t_conf:
#                         t_txt, t_conf = t2, c2
#             text, conf, engine = choose_reading(e_txt, e_conf, t_txt, t_conf)
#             alts = []
#             for a in (text, clean_text(t_txt), clean_text(e_txt)):
#                 if a and a not in alts:
#                     alts.append(a)
#             if not any(re.search(r"[^\W_]", a) for a in alts):
#                 continue
#             tokens.append(Token(text=text, alts=alts, conf=round(float(conf), 3),
#                                 engine=engine, box=[int(v) for v in box]))
#         return tokens


# --------------------------- PaddleOCR (active) ------------------------------
# One detector finds the text lines, then every line is read by two recognizers:
#   - arabic_PP-OCRv5 : Arabic + Latin + digits (the only one that can read Arabic)
#   - PP-OCRv6        : strongest on English / digits / emails
# The best reading becomes the token text, both are kept in `alts`.
_NON_AR_LA = re.compile(r"[^\u0000-\u007f؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿٠-٩]")


def choose_paddle_reading(a_txt, a_conf, e_txt, e_conf):
    a_txt, e_txt = clean_text(a_txt), clean_text(e_txt)
    a_ar, _ = script_counts(a_txt)
    # the latin model turns Arabic into junk like 'äenl li' / 'ス' / '☆'
    e_junk = bool(_NON_AR_LA.search(e_txt))
    if not e_txt or e_junk:
        return a_txt, a_conf, "paddle_ar"
    if not a_txt:
        return e_txt, e_conf, "paddle_en"
    if a_ar:                                      # Arabic seen -> only the Arabic model is trustworthy
        return a_txt, a_conf, "paddle_ar"
    if e_conf >= a_conf - PADDLE_EN_MARGIN:       # pure latin/digits -> prefer the latin model
        return e_txt, e_conf, "paddle_en"
    return a_txt, a_conf, "paddle_ar"


class PaddleHybridOCR:
    def __init__(self):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        import logging
        logging.disable(logging.WARNING)
        # paddleocr must be imported before paddle: paddleocr pulls in torch, and on Windows
        # torch fails to load its DLLs (WinError 127) if paddle was imported first
        from paddleocr import TextDetection, TextRecognition
        import paddle

        has_gpu = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        device = "gpu:0" if USE_GPU and has_gpu else "cpu"
        if USE_GPU and device == "cpu":
            print("[warn] no CUDA build of paddlepaddle or no GPU found -> running PaddleOCR on CPU")
        # enable_mkldnn=False works around a oneDNN crash in paddlepaddle 3.3 on CPU
        common = {"device": device, "enable_mkldnn": False}
        print(f"Loading PaddleOCR ({PADDLE_DET_MODEL} + {PADDLE_AR_REC_MODEL} + {PADDLE_EN_REC_MODEL}) on {device}...")
        self.det = TextDetection(model_name=PADDLE_DET_MODEL, **common)
        self.rec_ar = TextRecognition(model_name=PADDLE_AR_REC_MODEL, **common)
        self.rec_en = TextRecognition(model_name=PADDLE_EN_REC_MODEL, **common) if PADDLE_EN_REC_MODEL else None

    def detect(self, img):
        res = self.det.predict(img)
        H, W = img.shape[:2]
        boxes = []
        for poly in (res[0]["dt_polys"] if res else []):
            xs, ys = poly[:, 0], poly[:, 1]
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
        boxes = [[max(0, a), max(0, b), min(W, c), min(H, d)] for a, b, c, d in boxes]
        boxes = [b for b in boxes if b[2] - b[0] > 4 and b[3] - b[1] > 4]
        return merge_boxes(boxes) if PADDLE_MERGE_BOXES else boxes

    @staticmethod
    def _crop(img, box):
        x1, y1, x2, y2 = box
        H, W = img.shape[:2]
        p = max(2, int(PADDLE_CROP_PAD * (y2 - y1)))   # a little margin helps short words (من / إلى)
        return img[max(0, y1 - p):min(H, y2 + p), max(0, x1 - p):min(W, x2 + p)]

    @staticmethod
    def _rec(model, crops):
        if model is None or not crops:
            return [("", 0.0)] * len(crops)
        out = model.predict(crops, batch_size=PADDLE_BATCH_SIZE)
        return [(str(r["rec_text"]), float(r["rec_score"])) for r in out]

    def read(self, img):
        boxes = self.detect(img)
        crops = [self._crop(img, b) for b in boxes]
        ar = self._rec(self.rec_ar, crops)
        en = self._rec(self.rec_en, crops)
        tokens = []
        for box, (a_txt, a_conf), (e_txt, e_conf) in zip(boxes, ar, en):
            text, conf, engine = choose_paddle_reading(a_txt, a_conf, e_txt, e_conf)
            if conf < PADDLE_MIN_CONF:
                continue
            alts = []
            for a in (text, clean_text(a_txt), clean_text(e_txt)):
                if a and a not in alts and not _NON_AR_LA.search(a):
                    alts.append(a)
            if not alts or not any(re.search(r"[^\W_]", a) for a in alts):
                continue
            tokens.append(Token(text=text, alts=alts, conf=round(float(conf), 3),
                                engine=engine, box=[int(v) for v in box]))
        return tokens


# ================================== PARSER ==================================
def detect_provider(tokens):
    blob = " ".join(norm(a) for t in tokens for a in t.alts)
    best, best_score = "generic", 0
    for name, prof in PROVIDERS.items():
        score = sum(1 for kw in prof["detect"] if norm(kw) and norm(kw) in blob)
        if score > best_score:
            best, best_score = name, score
    return best


def build_label_map(provider):
    lm = {}
    for fld, labels in GENERIC_LABELS.items():
        for lab in labels:
            lm[norm(lab)] = fld
    for lab, fld in PROVIDERS.get(provider, {}).get("label_overrides", {}).items():
        lm[norm(lab)] = fld
    return lm


def match_label(tok, label_map):
    """returns (field, score, remainder_words) or None"""
    best = None
    for alt in tok.alts:
        words = clean_text(alt).split()
        full = norm(alt)
        if not words or not full:
            continue
        for lab, fld in label_map.items():
            cand = None
            if full == lab:
                cand = (100.0, [])
            else:
                nl = len(lab.split())
                if len(words) > nl:
                    if norm(" ".join(words[:nl])) == lab:
                        cand = (96.0, words[nl:])
                    elif norm(" ".join(words[-nl:])) == lab:
                        cand = (95.0, words[:-nl])
                if cand is None and len(lab) >= 5:
                    r = ratio(full, lab)
                    if r >= FUZZY_LABEL_THRESHOLD:
                        cand = (r, [])
            if cand and (best is None or (cand[0], len(lab)) > (best[1], best[3])):
                best = (fld, cand[0], cand[1], len(lab))
    return (best[0], best[1], best[2]) if best else None


def match_anchor(tok):
    """'From' / 'To Mobile Wallet' / 'إلى انستاباي' / 'من' -> (side, words_after, words_before)"""
    for alt in tok.alts:
        words = clean_text(alt).split()
        if not words or len(words) > 5:
            continue
        nw = [norm(w) for w in words]
        for side, keys in ANCHOR_WORDS.items():
            for i in {0, len(nw) - 1}:
                if nw[i] in keys:
                    return side, words[i + 1:], words[:i]
    return None


class ReceiptParser:
    def __init__(self, lines, tokens):
        self.lines = lines
        self.tokens = tokens
        self.warnings, self.derived = [], []
        self.label_lines = set()
        self.provider = detect_provider(tokens)
        self.label_map = build_label_map(self.provider)
        self.r = {"provider": self.provider, "status": None,
                  "sender": {k: None for k in EMPTY_PARTY},
                  "receiver": {k: None for k in EMPTY_PARTY},
                  "amount": None, "fees": None, "total": None, "currency": None,
                  "reference": None, "date": None, "date_iso": None, "note": None,
                  "other_phones": []}

    def parse(self):
        self._sentences()
        anchors = self._anchors()
        kv = self._key_values(anchors)
        self._parties(anchors, kv)
        self._simple_fields(kv)
        self._headline_amount()
        self._status_currency()
        self._validate_money()
        self._other_phones()
        self.r["derived"] = self.derived
        self.r["warnings"] = self.warnings
        return self.r

    # ---- a. sentence receipts ("You transferred 9 EGP To 010...") ----------
    def _sentences(self):
        pats = GENERIC_SENTENCES + PROVIDERS.get(self.provider, {}).get("sentences", [])
        for ln in self.lines:
            free = [t for t in ln.tokens if not t.role]
            if not free:
                continue
            cands = [ln.text(free)] + [a for t in free for a in t.alts]
            m = None
            for pat in pats:
                for c in cands:
                    m = re.search(pat, norm_soft(c))
                    if m:
                        break
                if m:
                    break
            if not m:
                continue
            g = m.groupdict()
            if g.get("amount") and self.r["amount"] is None:
                self.r["amount"] = parse_money([g["amount"]])
            for side in ("sender", "receiver"):
                if g.get(side + "_phone") and not self.r[side]["phone"]:
                    self.r[side]["phone"] = find_phone([g[side + "_phone"]])
            for t in free:
                t.role = "sentence"

    # ---- b. from / to anchors ----------------------------------------------
    def _anchors(self):
        found = []
        for ln in self.lines:
            for t in ln.tokens:
                if t.role:
                    continue
                a = match_anchor(t)
                if a:
                    side, after, before = a
                    t.role = "anchor_" + side
                    found.append({"side": side, "line": ln.idx, "token": t, "after": after, "before": before})
        return found

    # ---- c. label : value ----------------------------------------------------
    def _value_ok(self, fld, value):
        if not value:
            return False
        if fld in MONEY_FIELDS:
            return parse_money([value]) is not None
        if fld == "date":
            return parse_datetime(value) is not None
        if fld == "reference":
            return parse_reference([value]) is not None
        if fld.endswith(".phone"):
            return find_phone([value]) is not None
        return len(_AR_LET.findall(value) + _LA_LET.findall(value)) >= 2

    def _neighbour_value(self, ln, fld, anchor_lines):
        order = (-1, 1) if fld in MONEY_FIELDS else (1, -1)   # headline amount sits above its label
        for d in order:
            j = ln.idx + d
            if not 0 <= j < len(self.lines) or j in anchor_lines:
                continue
            nb = self.lines[j]
            if abs(nb.cy - ln.cy) > 3.0 * max(ln.h, nb.h):
                continue
            free = [t for t in nb.tokens if not t.role]
            if not free or any(match_label(t, self.label_map) for t in free):
                continue
            val = nb.text(free)
            if self._value_ok(fld, val):
                return val, free
        return None

    def _key_values(self, anchors):
        kv = {}
        anchor_lines = {a["line"] for a in anchors}
        for ln in self.lines:
            hits = []
            for t in ln.tokens:
                if t.role:
                    continue
                m = match_label(t, self.label_map)
                if m:
                    hits.append((t, m))
            if not hits:
                continue
            self.label_lines.add(ln.idx)
            for t, _ in hits:
                t.role = "label"
            others = [t for t in ln.tokens if not t.role]
            for t, (fld, score, rem) in hits:
                if len(hits) > 1:   # two columns on one line -> nearest label owns the value
                    mine = [o for o in others if min(hits, key=lambda h: abs(h[0].cx - o.cx))[0] is t]
                else:
                    mine = others
                value = " ".join(rem + ([ln.text(mine)] if mine else [])).strip()
                vtoks = list(mine)
                if not self._value_ok(fld, value):
                    nb = self._neighbour_value(ln, fld, anchor_lines)
                    if nb:
                        value, vtoks = nb
                for v in vtoks:
                    v.role = "value"
                if value:
                    kv.setdefault(fld, []).append({"value": value, "line": ln.idx, "tokens": vtoks})
        return kv

    # ---- sender / receiver blocks ---------------------------------------------
    def _read_party(self, toks, anchor, side):
        info = {k: None for k in EMPTY_PARTY}
        extra = []
        if anchor:
            if anchor["after"]:
                at = " ".join(anchor["after"])
                if find_phone([at]):
                    info["phone"] = find_phone([at])
                elif find_email([at]):
                    info["email"] = find_email([at])
                else:
                    info["account_type"] = at
            extra += anchor["before"]
        name_lines = {}
        at = anchor["token"] if anchor else None
        for t in toks:
            texts = [t.text] + t.alts
            em = find_email(texts)
            if em:
                info["email"] = info["email"] or em
                t.role = side + "_email"
                continue
            ph = find_phone(texts)
            if ph:
                info["phone"] = info["phone"] or ph
                t.role = side + "_phone"
                continue
            bank = find_bank(texts)
            if bank:
                info["bank"] = info["bank"] or bank
                t.role = side + "_bank"
                continue
            # bank logos / wallet icons ('BANQUE MISR', 'WLS') sit beside the text column,
            # completely outside the From/To anchor's x-range -> not the name
            if at and t.line != at.line and (t.x2 < at.x1 or t.x1 > at.x2):
                continue
            if is_name(clean_name(t.text)):
                name_lines.setdefault(t.line, []).append(t)
        info["bank"] = info["bank"] or find_bank(extra)
        names = []
        for li in sorted(name_lines):
            nm = clean_name(join_tokens(name_lines[li]))
            if is_name(nm) and nm not in names:
                names.append(nm)
            for t in name_lines[li]:
                t.role = side + "_name"
        if names:
            info["name"] = names[0]
        if len(names) > 1:
            info["name_alt"] = names[1]          # e.g. English + Arabic name of the same person
        return info

    def _parties(self, anchors, kv):
        stop = set(self.label_lines)
        first = {}
        for a in sorted(anchors, key=lambda a: a["line"]):
            first.setdefault(a["side"], a)
        anchor_lines = {a["line"] for a in first.values()}
        blocks = {}
        for side, a in first.items():
            idxs = [a["line"]]
            for j in range(a["line"] + 1, min(len(self.lines), a["line"] + 1 + MAX_PARTY_LINES)):
                if j in stop or j in anchor_lines:
                    break
                if self.lines[j].cy - self.lines[j - 1].cy > PARTY_GAP_RATIO * max(self.lines[j].h, self.lines[j - 1].h):
                    break
                idxs.append(j)
            blocks[side] = (idxs, a)
        # only 'to/إلى' found -> the block right above it is the sender
        if "receiver" in blocks and "sender" not in blocks:
            start = blocks["receiver"][1]["line"]
            idxs = []
            for j in range(start - 1, max(-1, start - 1 - MAX_PARTY_LINES), -1):
                if j in stop or j in anchor_lines or any(t.role == "sentence" for t in self.lines[j].tokens):
                    break
                idxs.insert(0, j)
            if idxs:
                blocks["sender"] = (idxs, None)
                self.warnings.append("sender block inferred from position (no 'From/من' found)")
        for side, (idxs, a) in blocks.items():
            toks = [t for j in idxs for t in self.lines[j].tokens if not t.role]
            info = self._read_party(toks, a, side)
            for k, v in info.items():
                if v and not self.r[side][k]:
                    self.r[side][k] = v
            for t in toks:
                t.role = t.role or side
        # explicit labels (اسم المستقبل / رقم المحفظة ...) win over block inference
        for fld, items in kv.items():
            if "." not in fld:
                continue
            side, key = fld.split(".")
            val = items[0]["value"]
            if key == "phone":
                val = find_phone([val] + [a for t in items[0]["tokens"] for a in t.alts]) or val
            elif key == "name":
                val = clean_name(val) or val
            self.r[side][key] = val

    # ---- simple fields -----------------------------------------------------------
    def _simple_fields(self, kv):
        amounts = kv.get("amount", [])
        if len(amounts) > 1 and "total" not in kv and "fees" in kv:
            later = [a for a in amounts[1:] if a["line"] > kv["fees"][0]["line"]]
            if later:
                kv["total"] = [later[0]]
                amounts.remove(later[0])
                self.warnings.append("second 'amount' label below fees treated as total")
        for fld in MONEY_FIELDS:
            if self.r[fld] is not None:
                continue
            for it in kv.get(fld, []):
                v = parse_money([it["value"]] + [a for t in it["tokens"] for a in t.alts])
                if v is not None:
                    self.r[fld] = v
                    break
        for it in kv.get("reference", []):
            ref = parse_reference([it["value"]] + [a for t in it["tokens"] for a in t.alts])
            if ref:
                self.r["reference"] = ref
                break
        for it in kv.get("date", []):
            for c in [it["value"]] + [a for t in it["tokens"] for a in t.alts]:
                iso = parse_datetime(c)
                if iso:
                    self.r["date"], self.r["date_iso"] = clean_text(c), iso
                    break
            if self.r["date_iso"]:
                break
        if not self.r["date_iso"]:                # no date label -> scan free text
            for ln in self.lines:
                txt = ln.text([t for t in ln.tokens if t.role not in ("label",)])
                iso = parse_datetime(txt)
                if iso:
                    self.r["date"], self.r["date_iso"] = txt, iso
                    break
        if kv.get("note"):
            self.r["note"] = clean_text(kv["note"][0]["value"])

    def _headline_amount(self):
        if self.r["amount"] is not None:
            return
        best = None
        for t in self.tokens:
            if t.role in ("label", "value"):
                continue
            n = " " + norm(" ".join(t.alts)) + " "
            if not any(c in n for c in (" egp ", " ج م ", "جنيه", " le ")):
                continue
            v = parse_money(t.alts)
            if v is not None and (best is None or t.h > best[0]):
                best = (t.h, v, t)
        if best:
            self.r["amount"] = best[1]
            best[2].role = best[2].role or "value"
            self.warnings.append("amount taken from the biggest EGP number (no label)")

    def _status_currency(self):
        blob = " " + " ".join(norm(a) for t in self.tokens for a in t.alts) + " "
        if any(norm(w) in blob for w in FAIL_WORDS):
            self.r["status"] = "failed"
        elif any(norm(w) in blob for w in SUCCESS_WORDS):
            self.r["status"] = "success"
        if any(c in blob for c in (" egp ", " ج م ", "جنيه")) or self.r["amount"] is not None:
            self.r["currency"] = "EGP"

    def _validate_money(self):
        a, f, t = self.r["amount"], self.r["fees"], self.r["total"]
        if a is not None and f is not None and t is not None:
            if abs(a + f - t) > 0.01:
                if abs(a - t) <= 0.01:
                    self.warnings.append(f"fees read as {f} but amount == total -> fees set to 0")
                    self.r["fees"] = 0.0
                else:
                    self.warnings.append("amount + fees != total, please check this receipt")
        elif a is not None and f is not None:
            self.r["total"] = round(a + f, 2)
            self.derived.append("total")
        elif a is not None and t is not None and t >= a:
            self.r["fees"] = round(t - a, 2)
            self.derived.append("fees")
        elif f is not None and t is not None:
            self.r["amount"] = round(t - f, 2)
            self.derived.append("amount")

    def _other_phones(self):
        assigned = {self.r["sender"]["phone"], self.r["receiver"]["phone"]}
        for t in self.tokens:
            if t.role in ("label", "value") or t.role.startswith("anchor"):
                continue
            ph = find_phone([t.text])
            if ph and ph not in assigned and ph not in self.r["other_phones"]:
                self.r["other_phones"].append(ph)


# ================================= I/O =====================================
def imread_any(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / max(1, img.max()))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:                      # transparent PNG -> white background
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3].astype(np.float32) * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
    return img


def imwrite_any(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if ok:
        buf.tofile(path)


def resize_for_ocr(img):
    long_side = max(img.shape[:2])
    s = 1.0
    if long_side < TARGET_LONG_SIDE:
        s = TARGET_LONG_SIDE / long_side
    elif long_side > MAX_LONG_SIDE:
        s = MAX_LONG_SIDE / long_side
    if s != 1.0:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    return img, s


def role_color(role):
    if role.startswith("anchor"):
        return (0, 0, 230)
    if role.startswith("sender"):
        return (230, 120, 0)
    if role.startswith("receiver"):
        return (0, 160, 0)
    return {"label": (140, 140, 140), "value": (0, 140, 255), "sentence": (200, 0, 200)}.get(role, (80, 80, 80))


def draw_debug(img, tokens, path):
    vis = img.copy()
    for t in tokens:
        c = role_color(t.role)
        cv2.rectangle(vis, (t.x1, t.y1), (t.x2, t.y2), c, 2)
        if t.role:
            cv2.putText(vis, t.role, (t.x1, max(12, t.y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    imwrite_any(path, vis)


def process_image(path, ocr):
    img0 = imread_any(path)
    img, scale = resize_for_ocr(img0)
    tokens = ocr.read(img)
    lines = group_lines(tokens)
    res = ReceiptParser(lines, tokens).parse()
    out = {"file": os.path.basename(path)}
    out.update(res)
    out["text_lines"] = [ln.text() for ln in lines]
    out["tokens"] = [{"text": t.text, "alts": t.alts, "conf": t.conf, "engine": t.engine, "role": t.role,
                      "line": t.line, "bbox": [int(round(v / scale)) for v in t.box]} for t in tokens]
    return out, img, tokens


CSV_COLUMNS = ["file", "provider", "status", "amount", "fees", "total", "currency", "reference",
               "date", "date_iso", "note",
               "sender_name", "sender_name_alt", "sender_phone", "sender_email", "sender_bank", "sender_account_type",
               "receiver_name", "receiver_name_alt", "receiver_phone", "receiver_email", "receiver_bank",
               "receiver_account_type", "other_phones", "derived", "warnings"]


def flatten(res):
    row = {k: res.get(k) for k in CSV_COLUMNS if k in res}
    for side in ("sender", "receiver"):
        for k in EMPTY_PARTY:
            row[f"{side}_{k}"] = res[side].get(k)
    for k in ("other_phones", "derived", "warnings"):
        row[k] = " | ".join(res.get(k) or [])
    return row


def main():
    files = sorted(p for p in glob.glob(os.path.join(INPUT_DIR, "*")) if p.lower().endswith(IMAGE_EXTENSIONS))
    if not files:
        print(f"No images found in {os.path.abspath(INPUT_DIR)}")
        return
    for sub in ("json", "text", "debug"):
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

    # ocr = HybridOCR()          # EasyOCR + Tesseract (old engine, commented out above)
    ocr = PaddleHybridOCR()
    results = []
    for i, path in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            res, img, tokens = process_image(path, ocr)
        except Exception as e:
            print(f"[{i}/{len(files)}] {name}: ERROR {e}")
            continue
        results.append(res)
        with open(os.path.join(OUTPUT_DIR, "json", name + ".json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        if SAVE_TEXT_FILES:
            with open(os.path.join(OUTPUT_DIR, "text", name + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(res["text_lines"]))
        if SAVE_DEBUG_IMAGES:
            draw_debug(img, tokens, os.path.join(OUTPUT_DIR, "debug", name + ".png"))
        print(f"[{i}/{len(files)}] {name}: {res['provider']} | amount={res['amount']} "
              f"ref={res['reference']} | from={res['sender']['name'] or res['sender']['phone']} "
              f"-> to={res['receiver']['name'] or res['receiver']['phone']}")

    summary = [{k: v for k, v in r.items() if k != "tokens"} for r in results]
    with open(os.path.join(OUTPUT_DIR, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, "results.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(flatten(r))
    print(f"\nDone. Results in {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()