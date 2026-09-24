"""Extract structured data from money-transfer receipt screenshots.

Layout-driven rather than template-driven, so new apps and layouts work without code:
  1. OCR lines keep their boxes; known OCR misreads are fixed (transaction_config.OCR_FIXES).
  2. Lines are recognised as labels from the vocabulary in transaction_config.py
     (exact, fuzzy for OCR typos, or "Label: value" in one line).
  3. "From" / "To" headers open a party block; its lines are classified by what they
     look like (phone, IPA handle / e-mail, bank + account, name).
  4. Every other label is paired with the value on its row, else just below it, else
     (for headline amounts captioned underneath) just above it. Values are validated
     against the field type, so a wrong neighbour is skipped.
  5. Leftovers are still typed and reported under "unmapped", so an unknown layout
     degrades gracefully instead of failing.

Usage: python transaction_extractor.py [image-or-folder ...]    (default: synthetic/)
"""
import difflib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import transaction_config as cfg

sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path(__file__).parent / "transaction_output"
DEFAULT_INPUT = Path(__file__).parent / "synthetic"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FUZZY_LABEL_RATIO = 0.85
MONEY_FIELDS = ("amount", "fees", "total")

# ---------------------------------------------------------------- text normalization

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬", "01234567890123456789.,")
ARABIC_LETTERS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
DIACRITICS = re.compile(r"[ً-ْـ]")
ARABIC = re.compile(r"[؀-ۿ]")


def clean(text):
    """Readable form used for values: ASCII digits, known OCR misreads fixed."""
    text = text.translate(ARABIC_DIGITS)
    for pattern, repl in cfg.OCR_FIXES:
        text = re.sub(pattern, repl, text)
    return " ".join(text.split())


def norm(text):
    """Comparison form used for labels and keywords."""
    text = DIACRITICS.sub("", text.lower()).translate(ARABIC_LETTERS)
    return " ".join(re.sub(r"[^\w&]+", " ", text).split())


def keyword_in(keyword, normalized_text, allow_inside_word=False):
    keyword = norm(keyword)
    if allow_inside_word and keyword.isascii() and len(keyword) >= 4:
        return keyword in normalized_text
    return re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", normalized_text) is not None

# ---------------------------------------------------------------- value parsers

PHONE = re.compile(r"(?:\+?20|0)1[0125]\d{8}")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*")
NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
CURRENCY_CODES = {t.lower(): code for code, tokens in cfg.CURRENCIES.items() for t in tokens}
CURRENCY = re.compile(
    r"(?<![A-Za-z؀-ۿ])(?:"
    + "|".join(map(re.escape, sorted(CURRENCY_CODES, key=len, reverse=True)))
    + r")(?![A-Za-z؀-ۿ])",
    re.IGNORECASE,
)
BANK = re.compile(
    r"^(" + "|".join(map(re.escape, sorted(cfg.BANKS, key=len, reverse=True))) + r")(?!\w)[\s:-]*(.*)$",
    re.IGNORECASE,
)
MONTHS = {
    **{m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)},
    **{m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                   "september", "october", "november", "december"], 1)},
    "sept": 9,
    **{norm(m): i for i, m in enumerate(["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس",
                                         "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"], 1)},
}
LATIN_MONTHS = [m for m in MONTHS if m.isascii()]
TIME = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AaPp]\.?[Mm]\b|ص|م)?")


def parse_money(text):
    """'22,000EGP' / 'EGP 501' / 'ج.م 1950' -> (22000, 'EGP'); None if it is not just an amount."""
    text = clean(text)
    currency = CURRENCY.search(text)
    rest = CURRENCY.sub(" ", text) if currency else text
    m = re.fullmatch(rf"\s*[-+]?\s*({NUMBER})\s*", rest)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    return (int(value) if value.is_integer() else value), (CURRENCY_CODES[currency.group().lower()] if currency else None)


def month_number(word):
    word = norm(word).rstrip(".")
    return MONTHS.get(word) or (MONTHS.get(word[:3]) if word.isascii() else None)


def fix_months(text):
    """OCR-damaged month names: 'SeD' -> 'sep'."""
    def repl(m):
        word = m.group().lower()
        if word in MONTHS or word in ("am", "pm"):
            return m.group()
        close = difflib.get_close_matches(word, LATIN_MONTHS, n=1, cutoff=0.66)
        return close[0] if close else m.group()
    return re.sub(r"[A-Za-z]{3,9}", repl, text)


def parse_datetime(text):
    """Any common receipt date (+ optional time) -> ISO string, day-first for numeric dates."""
    text = fix_months(clean(text))
    ymd = None
    for m in re.finditer(r"(?<!\d)(\d{1,2})\s*([^\W\d_]+)\.?,?\s*(\d{4})", text):
        if month := month_number(m.group(2)):
            ymd = (int(m.group(3)), month, int(m.group(1)))
            break
    if not ymd:
        for m in re.finditer(r"([^\W\d_]+)\.?\s+(\d{1,2}),?\s+(\d{4})", text):
            if month := month_number(m.group(1)):
                ymd = (int(m.group(3)), month, int(m.group(2)))
                break
    if not ymd and (m := re.search(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)", text)):
        ymd = tuple(map(int, m.groups()))
    if not ymd and (m := re.search(r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})(?!\d)", text)):
        day, month, year = map(int, m.groups())
        if month > 12:
            day, month = month, day
        ymd = (year + 2000 if year < 100 else year, month, day)
    if not ymd:
        return None

    hour = minute = second = 0
    if t := TIME.search(text):
        hour, minute, second = int(t.group(1)), int(t.group(2)), int(t.group(3) or 0)
        suffix = (t.group(4) or "").lower().replace(".", "")
        if suffix in ("pm", "م") and hour < 12:
            hour += 12
        elif suffix in ("am", "ص") and hour == 12:
            hour = 0
    try:
        dt = datetime(*ymd, hour, minute, second)
    except ValueError:
        return None
    return dt.isoformat() if t else dt.date().isoformat()


def classify(text):
    """What a free-standing value looks like -> [(slot, value), ...]."""
    text = clean(text)
    compact = re.sub(r"[\s-]", "", text)
    if m := EMAIL.search(text):
        return [("account", m.group())]            # e-mail or InstaPay address (name@instapay)
    if PHONE.fullmatch(compact):
        return [("phone", compact)]
    if (m := BANK.match(text)) and re.search(r"\d", m.group(2)):
        return [("bank", m.group(1)), ("account", m.group(2))]
    if m := re.fullmatch(r"([A-Z]{2,6})\s+(\d[\dA-Z*-]{4,})", text):
        return [("bank", m.group(1)), ("account", m.group(2))]
    if re.fullmatch(r"[\dA-Z*xX#]{6,}", compact) and sum(c.isdigit() for c in compact) >= 4:
        return [("account", text)]                 # IBAN, card, masked or plain account number
    if len(re.findall(r"[^\W\d_]", text)) >= 3:
        return [("name", text)]
    return []


def valid(target, text):
    slot = target.split(".")[-1]
    if slot in MONEY_FIELDS:
        return parse_money(text) is not None
    if slot == "datetime":
        return parse_datetime(text) is not None
    if slot == "reference":
        compact = text.replace(" ", "")
        return re.fullmatch(r"[\w\-/#.]{4,}", compact) is not None and any(c.isdigit() for c in compact)
    if slot == "account":
        return bool(classify(text))
    if slot == "name":
        return len(re.findall(r"[^\W\d_]", text)) >= 2
    return re.search(r"\w", text) is not None

# ---------------------------------------------------------------- labels

LABELS = sorted(
    [(norm(w), "field", target) for target, words in cfg.FIELD_LABELS.items() for w in words]
    + [(norm(w), "party", party) for party, words in cfg.PARTY_HEADERS.items() for w in words],
    key=lambda entry: -len(entry[0]),  # longest first, so "رقم المرسل إليه" wins over "رقم المرسل"
)


def match_label(text):
    """(kind, target, inline_value) if the line is a label, else None."""
    n = norm(text)
    if not n:
        return None
    for syn, kind, target in LABELS:
        if n == syn:
            return kind, target, ""

    if len(n) >= 4:
        score, kind, target = max(
            ((difflib.SequenceMatcher(None, n, syn).ratio(), kind, target) for syn, kind, target in LABELS if len(syn) >= 4),
            key=lambda e: e[0],
        )
        if score >= FUZZY_LABEL_RATIO:
            return kind, target, ""

    # "Date: 23 Sep 2026", "To Mobile Wallet", "إلى انستاباي" - label and value in one OCR line
    words = text.split()
    for syn, kind, target in LABELS:
        for k in range(1, len(words)):
            for label_part, rest in ((words[:k], words[k:]), (words[-k:], words[:-k])):
                if norm(" ".join(label_part)) == syn:
                    value = " ".join(rest)
                    if kind == "party" or valid(target, value):
                        return kind, target, value
    return None

# ---------------------------------------------------------------- geometry

def height(line):
    return line["box"][3] - line["box"][1]


def v_overlap(a, b):
    """Shared height as a fraction of the shorter box."""
    return max(0, min(a[3], b[3]) - max(a[1], b[1])) / max(1, min(a[3] - a[1], b[3] - b[1]))


def h_overlap(a, b, tol=0):
    return a[0] - tol < b[2] and b[0] - tol < a[2]


def h_gap(a, b):
    return max(a[0], b[0]) - min(a[2], b[2])


def merge_fragments(lines):
    """Join value pieces the OCR split within one phrase, e.g. headline '45' + 'EGP'."""
    merged = []
    for line in sorted(lines, key=lambda l: l["box"][0]):
        for m in merged:
            if (not m["label"] and not line["label"] and v_overlap(m["box"], line["box"]) > 0.6
                    and h_gap(m["box"], line["box"]) < 0.5 * min(height(m), height(line))):
                rtl = ARABIC.search(m["text"] + line["text"])
                m["text"] = f"{line['text']} {m['text']}" if rtl else f"{m['text']} {line['text']}"
                m["box"] = [min(m["box"][0], line["box"][0]), min(m["box"][1], line["box"][1]),
                            max(m["box"][2], line["box"][2]), max(m["box"][3], line["box"][3])]
                m["label"] = match_label(m["text"])
                break
        else:
            merged.append(line)
    return sorted(merged, key=lambda l: (l["box"][1], l["box"][0]))


def prepare(ocr_lines):
    lines = []
    for raw in ocr_lines:
        text = clean(raw["text"])
        if text:
            lines.append({"text": text, "box": list(raw["box"]), "label": match_label(text)})
    return merge_fragments(lines)

# ---------------------------------------------------------------- extraction

def party_section(lines, i):
    """Indices of the lines under a From/To header: same column, no big gap, up to the next label."""
    head = lines[i]["box"]
    h = head[3] - head[1]
    section, bottom = [], head[3]
    for j in range(i + 1, len(lines)):
        line = lines[j]
        if line["box"][1] < head[1] + 0.5 * h:              # same row as the header (icons etc.)
            continue
        if line["label"]:
            break
        if not h_overlap(head, line["box"]) or len(line["text"]) < 3:
            continue                                          # logos / icons beside the column
        if line["box"][1] - bottom > 2.5 * max(h, height(line)):
            break
        section.append(j)
        bottom = line["box"][3]
    return section


def find_value(lines, i, target, used):
    """Value paired with label i: same row, else just below, else (amounts only) just above."""
    label = lines[i]["box"]
    h = label[3] - label[1]
    free = [j for j, line in enumerate(lines) if j not in used and not line["label"]]
    same_row = sorted((j for j in free if v_overlap(label, lines[j]["box"]) >= 0.5),
                      key=lambda j: h_gap(label, lines[j]["box"]))
    below = sorted((j for j in free if j not in same_row and h_overlap(label, lines[j]["box"])
                    and -0.3 * h <= lines[j]["box"][1] - label[3] <= 2 * h),
                   key=lambda j: lines[j]["box"][1])
    above = []
    if target in MONEY_FIELDS:  # headline amount with its caption underneath ("EGP 500" / "Total amount")
        above = sorted((j for j in free if j not in same_row and h_overlap(label, lines[j]["box"])
                        and -0.3 * h <= label[1] - lines[j]["box"][3] <= 2 * h),
                       key=lambda j: -lines[j]["box"][3])
    for j in same_row + below + above:
        if valid(target, lines[j]["text"]):
            used.add(j)
            return lines[j]["text"]
    return None


def add_party(party, pairs):
    for slot, value in pairs:
        if slot in party:
            party.setdefault("other", []).append(value)
        else:
            party[slot] = value


def header_value(text):
    """Text after a party header is either a channel ('Mobile Wallet') or a party value."""
    if any(norm(text) == norm(c) for c in cfg.CHANNELS):
        return [("via", text)]
    return classify(text)


def set_field(result, target, value):
    if target in MONEY_FIELDS:
        if result[target] is None:
            result[target], currency = parse_money(value)
            result["currency"] = result["currency"] or currency
    elif target == "datetime":
        result["datetime"] = result["datetime"] or parse_datetime(value)
    elif "." in target:
        party, slot = target.split(".")
        add_party(result[party], classify(value) if slot == "account" else [(slot, value)])
    elif result[target] is None:
        result[target] = value


def describe(text):
    """Type of a leftover line, or None if it is just text."""
    money = parse_money(text)
    if money and money[1]:
        return "money"
    if parse_datetime(text):
        return "datetime"
    kinds = [slot for slot, _ in classify(text) if slot != "name"]
    return kinds[-1] if kinds else None


def extract(ocr_lines):
    lines = prepare(ocr_lines)
    result = {
        "provider": None, "status": None, "type": None,
        "amount": None, "fees": None, "total": None, "currency": None,
        "sender": {}, "receiver": {},
        "reference": None, "datetime": None, "note": None,
        "evidence": [], "unmapped": [],
    }
    used = set()

    # 1. From / To blocks
    for i, line in enumerate(lines):
        if line["label"] and line["label"][0] == "party":
            _, party, inline = line["label"]
            used.add(i)
            if inline:
                add_party(result[party], header_value(inline))
            for j in party_section(lines, i):
                used.add(j)
                add_party(result[party], classify(lines[j]["text"]))
                result["evidence"].append({"field": party, "label": line["text"], "value": lines[j]["text"]})

    # 2. label -> value pairs
    for i, line in enumerate(lines):
        if line["label"] and line["label"][0] == "field":
            _, target, inline = line["label"]
            used.add(i)
            value = inline or find_value(lines, i, target, used)
            if value is not None:
                set_field(result, target, value)
                result["evidence"].append({"field": target, "label": line["text"], "value": value})

    # 3. sender block whose "From" header the OCR missed: unlabeled lines right above the "To" header
    receiver_header = next((i for i, l in enumerate(lines) if l["label"] and l["label"][:2] == ("party", "receiver")), None)
    if not result["sender"] and receiver_header is not None:
        head = lines[receiver_header]["box"]
        for j in range(receiver_header - 1, -1, -1):
            if j in used or lines[j]["label"]:
                break
            if len(lines[j]["text"]) >= 3 and h_overlap(head, lines[j]["box"]):
                used.add(j)
                add_party(result["sender"], classify(lines[j]["text"]))
                result["evidence"].append({"field": "sender", "label": None, "value": lines[j]["text"]})

    # 4. headline amount without a recognised caption: the biggest amount with a currency
    if result["amount"] is None:
        candidates = [j for j, l in enumerate(lines) if j not in used and not l["label"]
                      and (m := parse_money(l["text"])) and m[1]]
        if candidates:
            j = max(candidates, key=lambda j: height(lines[j]))
            used.add(j)
            set_field(result, "amount", lines[j]["text"])
            result["evidence"].append({"field": "amount", "label": None, "value": lines[j]["text"]})

    # 5. leftovers
    for j, line in enumerate(lines):
        if j not in used and not line["label"] and len(line["text"]) >= 3:
            kind = describe(line["text"])
            result["unmapped"].append({"text": line["text"], "type": kind})
            if kind == "datetime" and result["datetime"] is None:
                result["datetime"] = parse_datetime(line["text"])

    leftover_text = norm(" ".join(u["text"] for u in result["unmapped"]))
    result["status"] = next((status for status, words in cfg.STATUS_KEYWORDS.items()
                             if any(keyword_in(w, leftover_text) for w in words)), None)
    all_text = norm(" ".join(l["text"] for l in lines))
    result["provider"] = next((name for name, words in cfg.PROVIDERS.items()
                               if any(keyword_in(w, all_text, allow_inside_word=True) for w in words)), None)
    return result, [l["text"] for l in lines]

# ---------------------------------------------------------------- CLI

def iter_images(args):
    for path in map(Path, args or [DEFAULT_INPUT]):
        if path.is_dir():
            yield from sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        elif path.is_file():
            yield path
        else:
            print(f"Skipping missing path: {path}", file=sys.stderr)


def print_summary(name, result):
    print(f"--- {name} ---")
    for key in ("provider", "status", "type", "amount", "fees", "total", "currency", "reference", "datetime", "note"):
        if result[key] is not None:
            print(f"  {key:<18} {result[key]}")
    for party in ("sender", "receiver"):
        for slot, value in result[party].items():
            print(f"  {party + '.' + slot:<18} {value}")
    for item in result["unmapped"]:
        if item["type"]:
            print(f"  {'unmapped.' + item['type']:<18} {item['text']}")
    print()


def main():
    images = list(iter_images(sys.argv[1:]))
    if not images:
        sys.exit("No images found.")

    from paddle_ocr import create_ocr, extract_lines  # heavy import, only when actually running OCR
    ocr = create_ocr()
    OUTPUT_DIR.mkdir(exist_ok=True)

    for image_path in images:
        result, lines = extract(extract_lines(ocr, image_path))
        print_summary(image_path.name, result)
        output_file = OUTPUT_DIR / f"{image_path.stem}.json"
        output_file.write_text(
            json.dumps({"image": str(image_path), **result, "ocr_lines": lines}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"Saved {len(images)} result(s) to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
