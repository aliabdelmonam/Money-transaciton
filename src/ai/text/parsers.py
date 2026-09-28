"""Value parsers: phone, email, bank, money, reference, date, names.

Every `find_*` / `parse_*` takes a list of candidate readings and returns the
first hit, so callers can pass all OCR alternatives of a token.
"""

import difflib
import re
from datetime import datetime
from typing import Iterable, Optional

from ..knowledge import BANKS, NOT_NAMES
from .normalize import AR_LETTER, LATIN_LETTER, clean_text, norm, norm_soft, to_ascii_digits

_MONEY_RE = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?(?!\d)")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?20|0020)?0?(1[0125]\d{8})(?!\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+\s?@\s?[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*")

_NOT_NAMES = {norm(x) for x in NOT_NAMES}
_BANK_KEYS = [(" " + norm(k) + " ", v) for k, v in BANKS.items()]


def find_phone(texts: Iterable[str]) -> Optional[str]:
    for t in texts:
        s = to_ascii_digits(clean_text(t))
        for cand in (s, re.sub(r"[\s\-().]", "", s)):
            m = PHONE_RE.search(cand)
            if m:
                return "0" + m.group(1)
    return None


def find_email(texts: Iterable[str]) -> Optional[str]:
    for t in texts:
        m = EMAIL_RE.search(clean_text(t))
        if m:
            return re.sub(r"\s", "", m.group(0)).lower()
    return None


def find_bank(texts: Iterable[str]) -> Optional[str]:
    for t in texts:
        n = " " + norm(t) + " "
        for key, name in _BANK_KEYS:
            if key in n:
                return name
    return None


def parse_money(texts: Iterable[str]) -> Optional[float]:
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
        s = re.sub(r"(?<=[A-Za-z؀-ۿ])(?=\d)|(?<=\d)(?=[A-Za-z؀-ۿ])", " ", s)
        best = None
        for m in _MONEY_RE.finditer(s):
            whole = m.group(1).replace(",", "")
            val = float(whole + ("." + m.group(2) if m.group(2) else ""))
            if best is None or len(whole) > best[0]:
                best = (len(whole), val)
        if best:
            return best[1]
    return None


def parse_reference(texts: Iterable[str]) -> Optional[str]:
    for t in texts:
        s = to_ascii_digits(clean_text(t))
        runs = [r.strip("-") for r in re.findall(r"[A-Za-z0-9\-]{4,}", s)]
        runs = [r for r in runs if len(r) >= 4]
        if runs:
            return max(runs, key=len)
    return None


# ---------------------------------- dates ----------------------------------
_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
           "sep": 9, "oct": 10, "nov": 11, "dec": 12,
           "يناير": 1, "فبراير": 2, "مارس": 3, "ابريل": 4, "مايو": 5, "يونيو": 6, "يوليو": 7,
           "اغسطس": 8, "سبتمبر": 9, "اكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12}
_OCR_FIX = str.maketrans({"v": "y", "0": "o", "1": "l", "5": "s", "8": "b"})


def month_num(word: str) -> Optional[int]:
    w = norm(word)
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


def _find_date(s: str):
    """-> (year, month, day, span) or None"""
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?[\s\-/.,]*([a-zء-ي]{3,9})\.?[\s\-/.,]*(\d{4})", s)
    if m and month_num(m.group(2)):
        return int(m.group(3)), month_num(m.group(2)), int(m.group(1)), m.span()
    m = re.search(r"([a-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})", s)
    if m and month_num(m.group(1)):
        return int(m.group(3)), month_num(m.group(1)), int(m.group(2)), m.span()
    m = re.search(r"(\d{4})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{1,2})", s)
    if m:
        y, mo, d = map(int, m.groups())
        return y, mo, d, m.span()
    m = re.search(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})", s)
    if m:
        d, mo, y = map(int, m.groups())
        if y < 100:
            y += 2000
        if mo > 12 and d <= 12:
            d, mo = mo, d
        return y, mo, d, m.span()
    return None


def _find_time(s: str):
    """-> (hour, minute) or None"""
    m = re.search(r"(\d{1,2})\s*[:.]\s*(\d{2})(?:\s*[:.]\s*\d{2})?\s*(a\.?\s?m|p\.?\s?m|p[1l|i]|ص|م)?", s)
    if not m:
        return None
    hh, mi = int(m.group(1)), int(m.group(2))
    ampm = (m.group(3) or "").replace(".", "").replace(" ", "")
    if (ampm.startswith("p") or ampm == "م") and hh < 12:
        hh += 12
    elif (ampm.startswith("a") or ampm == "ص") and hh == 12:
        hh = 0
    return (hh, mi) if 0 <= hh <= 23 and 0 <= mi <= 59 else None


def parse_datetime(text: str) -> Optional[str]:
    """ISO string ('2026-09-19T13:37' or '2026-09-19') or None. Date part required."""
    s = norm_soft(text)
    found = _find_date(s)
    if not found:
        return None
    y, mo, d, span = found
    if not (1 <= mo <= 12 and 1 <= d <= 31 and 1990 <= y <= 2100):
        return None
    tm = _find_time(s[:span[0]] + " " + s[span[1]:])
    try:
        dt = datetime(y, mo, d, *(tm or (0, 0)))
    except ValueError:
        return None
    return dt.isoformat(timespec="minutes") if tm else dt.date().isoformat()


# ---------------------------------- names ----------------------------------
def clean_name(s: str) -> str:
    s = clean_text(s)
    s = re.sub(r"[^\w\s*.\-']", " ", s)
    s = re.sub(r"[\d_]", " ", s)
    return re.sub(r"\s+", " ", s).strip(" .-'")


def is_name(s: str) -> bool:
    if not s or norm(s) in _NOT_NAMES:
        return False
    letters = AR_LETTER.findall(s) + LATIN_LETTER.findall(s)
    if len(letters) < 3:
        return False
    return len({c.lower() for c in letters}) > 1       # rejects 'nnn' logo garbage


def letter_count(s: str) -> int:
    return len(AR_LETTER.findall(s or "")) + len(LATIN_LETTER.findall(s or ""))
