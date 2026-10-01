"""Normalisers that turn raw OCR text into a canonical value, or None when the text can't
be a valid value of that kind. They are the vote keys of ``reader.LineReader.read_voted``,
so OCR variants of one value collapse together and an invalid reading never wins.
"""
import re
from datetime import datetime
from difflib import SequenceMatcher

# Characters OCR confuses with digits inside numeric fields.
DIGIT_FIXES = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1",
                             "i": "1", "Z": "2", "S": "5", "s": "5", "B": "8", "g": "9", "q": "9"})
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
AR_MONTHS = ["يناير", "فبراير", "مارس", "ابريل", "مايو", "يونيو", "يوليو", "اغسطس", "سبتمبر", "اكتوبر", "نوفمبر", "ديسمبر"]
CURRENCIES = ["EGP", "USD", "EUR", "SAR", "AED", "KWD", "GBP"]
# Letters OCR reads in place of a currency-code letter ("ECP", "ESP", "EGF", "USO").
CURRENCY_SLIPS = {"G": "CSOQ", "P": "FR", "D": "OQ", "U": "V"}
# E-mail / InstaPay address domains that OCR garbles ("instapav", "lnstapay").
KNOWN_DOMAINS = ["instapay"]


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def digits(text, min_len=1, max_len=40):
    """Digit-only value (phone, account, reference numbers)."""
    t = re.sub(r"[\s\-]", "", text).translate(DIGIT_FIXES)
    return t if re.fullmatch(r"\d{%d,%d}" % (min_len, max_len), t) else None


def code(text):
    """A reference code of letters and digits in one token (``44bf3c8``, ``TX9A27K1``):
    at least two digits and a letter, so neither a word nor a plain number."""
    t = text.strip()
    if not re.fullmatch(r"[A-Za-z0-9]{5,24}", t) or len(re.findall(r"\d", t)) < 2 or not re.search("[A-Za-z]", t):
        return None
    return t


def amount(text):
    """``1,500.50`` -> ``1500.50``; None unless the thousands separators are well placed
    ("50,00" or "5,0000" is a misread, not 5000 or 50000) and the value isn't 0."""
    t = text.replace(" ", "").translate(DIGIT_FIXES)
    m = re.search(r"\d[\d,]*(?:\.\d+)?", t)
    # ".5" lost its leading digit(s): 0.5? 10.5? Not a value to guess.
    if not m or t[m.start() - 1:m.start()] == ".":
        return None
    number = m.group(0).rstrip(",")
    if not (re.fullmatch(r"\d+(\.\d{1,2})?", number) or re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d{1,2})?", number)):
        return None
    value = number.replace(",", "")
    return value if float(value) != 0 else None


def currency(text):
    """Currency code; OCR slips like ECP / ESP / EGF still resolve to EGP, but a fragment
    ("E", "EP"), a tie between codes or a hop to another code ("ECR" is not EUR) is None."""
    t = re.sub(r"[^A-Z]", "", text.upper())
    if len(t) < 3:
        return None
    if t in CURRENCIES:
        return t
    if len(t) == 3:
        # one wrong letter, and only a look-alike of the real one
        hits = [c for c in CURRENCIES
                if sum(a != b for a, b in zip(c, t)) == 1
                and all(a == b or b in CURRENCY_SLIPS.get(a, "") for a, b in zip(c, t))]
        return hits[0] if len(hits) == 1 else None
    (best_score, best), (second_score, _) = sorted(((similar(c, t), c) for c in CURRENCIES), reverse=True)[:2]
    return best if best_score >= 0.6 and best_score > second_score and best[0] == t[0] else None


def email(text):
    """E-mail or InstaPay address (``muhammedd002@instapay``), a garbled known domain fixed."""
    t = text.lower().replace(" ", "")
    m = re.search(r"([a-z0-9._\-]+)@([a-z0-9]+(?:\.[a-z0-9]+)*)", t)
    if not m:
        return None
    user, domain = m.group(1).strip("._-"), m.group(2)
    if not user:
        return None
    for known in KNOWN_DOMAINS:
        if "." not in domain and similar(domain, known) >= 0.7:
            domain = known
    return f"{user}@{domain}"


AMPM = r"([AaPp][MmNn]|ص|م)"
NAMED_DATE = re.compile(r"([0-9OoIl]{1,2})[\s.,]*([A-Za-z؀-ۿ]{3,9})\.?\s*([0-9OoIlZSB]{4})\s*[,\-|]?\s*"
                        r"([0-9OoIl]{1,2})\s*[:.]\s*([0-9OoIlSB]{2})\s*" + AMPM + "?")
# "22/09/2026 | 10:11 PM": the separator bar is often read as a digit ("110:11"), so it may
# eat one digit, but only when the time doesn't parse without it
NUMERIC_DATE = re.compile(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})\s*[|Il1]??\s*(\d{1,2})\s*[:.]\s*(\d{2})\s*" + AMPM + "?")
TIME_FIRST = re.compile(r"(\d{1,2})\s*[:.]\s*(\d{2})\s*" + AMPM + r"[\s.,]*(\d{1,2})[\s.,]*([A-Za-z؀-ۿ]{3,9})\.?\s*(\d{4})")


def date(text):
    """A receipt date + time in one canonical form, so OCR variants vote together:
    ``20 Sep 2026 6:30 pm`` / ``04:26 PM. 19 Sep 2025`` -> ``20 Sep 2026 06:30 PM``
    (Arabic month names too), ``22/09/2026 | 10:11 PM`` -> ``22/09/2026 10:11 PM``.
    A named-month date needs its AM/PM marker: those receipts print 12-hour times, and
    without it the hour is ambiguous."""
    text = text.strip()
    if m := NAMED_DATE.search(text):
        day, month_s, year, hour, minute, ampm = m.groups()
    elif m := TIME_FIRST.search(text):
        hour, minute, ampm, day, month_s, year = m.groups()
    elif m := NUMERIC_DATE.search(text):
        day, month, year, hour, minute, ampm = m.groups()
        h, mi = int(hour), int(minute)
        if not (1 <= int(day) <= 31 and 1 <= int(month) <= 12 and mi < 60 and (h <= 12 if ampm else h < 24)):
            return None
        suffix = f" {_ampm(ampm)}" if ampm else ""
        return f"{int(day):02d}/{int(month):02d}/{year} {h:02d}:{mi:02d}{suffix}"
    else:
        return None
    if not ampm:
        return None
    day, year, hour, minute = (x.translate(DIGIT_FIXES) for x in (day, year, hour, minute))
    month = _month(month_s)
    h = int(hour)
    if month is None or not 1 <= h <= 12:
        return None
    h = h % 12 + (12 if _ampm(ampm) == "PM" else 0)
    try:
        d = datetime(int(year), month, int(day), h, int(minute))
    except ValueError:
        return None
    # built by hand rather than strftime("%b %p"), which follows the OS locale
    return (f"{d.day:02d} {MONTHS[d.month - 1].title()} {d.year} "
            f"{(d.hour - 1) % 12 + 1:02d}:{d.minute:02d} {'PM' if d.hour >= 12 else 'AM'}")


def _ampm(marker):
    return "PM" if marker.lower().startswith("p") or marker == "م" else "AM"


def _month(s):
    s = s.lower()
    if re.search(r"[؀-ۿ]", s):
        s = s.replace("أ", "ا").replace("إ", "ا")
        scores = [similar(s, m) for m in AR_MONTHS]
    else:
        s = s[:3].replace("0", "o").replace("1", "l")
        scores = [similar(s, m) for m in MONTHS]
    best = max(range(12), key=lambda i: scores[i])
    return best + 1 if scores[best] >= 0.6 else None
