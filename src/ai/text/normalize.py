"""Arabic/English text normalization used for matching."""

import difflib
import re
import unicodedata

try:
    from rapidfuzz import fuzz as _fuzz

    def ratio(a: str, b: str) -> float:
        return float(_fuzz.ratio(a, b))
except ImportError:
    def ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio() * 100.0

_BIDI = re.compile(r"[​-‏‪-‮⁦-⁩﻿]")
_DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
AR_LETTER = re.compile(r"[ء-غف-يٱ-ۓ]")
LATIN_LETTER = re.compile(r"[A-Za-z]")
# anything outside ASCII / Arabic blocks (latin models turn Arabic into 'ス', '☆', ...)
NON_AR_LATIN = re.compile(r"[^\u0000-\u007f؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿٠-٩]")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬", "01234567890123456789.,")


def clean_text(s) -> str:
    s = unicodedata.normalize("NFKC", _BIDI.sub("", str(s or "")))
    s = re.sub(r"\s+", " ", s)
    return s.strip(" -_|،;:")


def to_ascii_digits(s: str) -> str:
    return (s or "").translate(_DIGITS)


def _ar_norm(s: str) -> str:
    s = _DIAC.sub("", s)
    s = re.sub("[إأآٱ]", "ا", s)
    return s.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")


def norm(s: str) -> str:
    """Aggressive normalization for matching (no punctuation)."""
    s = _ar_norm(to_ascii_digits(clean_text(s)).lower())
    s = re.sub(r"[^\w@*]+", " ", s)
    return s.strip()


def norm_soft(s: str) -> str:
    """Normalization that keeps punctuation (numbers, sentences)."""
    return _ar_norm(to_ascii_digits(clean_text(s)).lower())


def script_counts(s: str) -> tuple[int, int]:
    """(arabic letters, latin letters)"""
    return len(AR_LETTER.findall(s or "")), len(LATIN_LETTER.findall(s or ""))


def has_word_char(s: str) -> bool:
    return bool(re.search(r"[^\W_]", s or ""))
