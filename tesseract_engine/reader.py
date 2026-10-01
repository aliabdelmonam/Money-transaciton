"""Line-level OCR with Tesseract.

Each block is cropped, scaled to a comfortable text height, contrast-stretched (so a
light-grey label is as crisp as a black value) and read on its own. The first pass reads
every line with the English and the Arabic model in raw-line mode (psm 13, which keeps
short and light lines) and keeps the more plausible script. Lines that need more care are
then re-read with their own strategy: digits / addresses / dates voted across scales,
masked names rebuilt around their asterisks, the headline amount split by colour, and
mixed Arabic + English lines read with the combined model.
"""
import re
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

from . import fields

TARGET_HEIGHT = 56  # px of block height fed to tesseract
PAD = 6
BORDER = 24  # white frame around every crop: tesseract misreads glyphs touching the edge

ARABIC = re.compile("[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")
ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
# bidi marks the Arabic model emits around numbers and Latin words
BIDI = re.compile("[‎‏؜‪-‮⁦-⁩]")
# OCR noise at the edges of a crop (quotes, bars, dashes, dots)
EDGE_NOISE = "-_.,:;'\"`‘’“”|!()[]{}~ـ،؛"
LOWER_ALNUM = "abcdefghijklmnopqrstuvwxyz0123456789"
ALNUM = LOWER_ALNUM + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GAMMA = 2.0
GAMMA_LUT = (255 * (np.arange(256) / 255.0) ** GAMMA).round().astype(np.uint8)
DIGITS = re.compile(r"\d")
# A reading with fewer characters than this share of what fits in its line is suspect
# (the Arabic model reads "To Instapay" as "اا").
MIN_FILL = 0.3
# A word shorter than this x the line's tallest word was misread at the line's scale
# ("You transferred" before a big "9 EGP"): it is read again on its own.
SMALL_WORD = 0.75
# Initials of masked names are voted at these sizes (default size first), on a crop a few
# px wider than the letter (Arabic dots and tails reach past its columns), until
# LETTER_QUORUM reads agree.
LETTER_HEIGHTS = (56, 48, 64, 40, 72)
LETTER_PAD = 3
LETTER_QUORUM = 3


@dataclass
class Reading:
    text: str
    conf: float  # 0-100

    def __bool__(self):
        return bool(self.text)


def clean(text):
    text = BIDI.sub("", text.translate(ARABIC_INDIC))
    return re.sub(r"\s+", " ", text).strip().strip(EDGE_NOISE + " ")


def has_arabic(text):
    return bool(ARABIC.search(text))


def has_latin(text):
    return bool(LATIN.search(text))


def crop_scale(b, target_height=TARGET_HEIGHT):
    """Factor that brings a block to ``target_height``: small text is enlarged and large
    text (the amount) shrunk, so every read sees text at the size tesseract handles best."""
    return float(np.clip(target_height / max(b.h, 1), 0.4, 5.0))


def crop(gray, b, target_height=TARGET_HEIGHT, x0=None, x1=None):
    h, w = gray.shape
    x0 = b.x if x0 is None else x0
    x1 = b.x2 if x1 is None else x1
    c = gray[max(0, b.y - PAD):min(h, b.y2 + PAD), max(0, x0 - PAD):min(w, x1 + PAD)]
    scale = crop_scale(b, target_height)
    if scale != 1.0:
        c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    # stretch contrast so light-grey labels become as crisp as black values, then darken
    # mid greys: a grey word beside black ones ("To" in "To Mobile Wallet") is otherwise
    # binarised away with the page
    c = cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX)
    c = GAMMA_LUT[c]
    return cv2.copyMakeBorder(c, BORDER, BORDER, BORDER, BORDER, cv2.BORDER_CONSTANT, value=255)


def pick_script(eng, ara, width_chars=None):
    """Choose between the English and the Arabic reading of one line.

    Each model reads the other's script as garbage in its own, so a reading counts half
    when it has no letters (or digits) of its own script, or when it is far too short for
    the line (``width_chars``: about how many characters fit in it). The Arabic one also
    counts half when it lost the digits of a mostly-digit English reading ("EGP501" read
    as "لشفت"): the Arabic model reads digits too. Not the other way round: the Arabic
    model turns Latin letters into digits, and the English one Arabic letters. For the
    same reason an Arabic reading of digits only, without a single Arabic letter, counts
    only as digits beside an English reading with Latin letters ("EGP 1" read as "601").
    """
    def score(r, own_script):
        s = r.conf * (1.0 if own_script else 0.5)
        if width_chars and len(r.text.replace(" ", "")) < MIN_FILL * width_chars:
            s *= 0.5
        return s

    eng_chars = eng.text.replace(" ", "")
    eng_digits = len(DIGITS.findall(eng_chars))
    lost_digits = eng_digits >= 2 and eng_digits >= 0.5 * len(eng_chars) and not DIGITS.search(ara.text)
    latin_as_digits = not has_arabic(ara.text) and has_latin(eng.text)
    ara_score = score(ara, has_arabic(ara.text) or DIGITS.search(ara.text)) * (0.5 if lost_digits or latin_as_digits else 1)
    eng_score = score(eng, has_latin(eng.text) or eng_digits)
    return (ara, "ara") if ara_score > eng_score else (eng, "eng")


def ink_mask(gray_region):
    _, m = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m > 0


class LineReader:
    def __init__(self, engines=None, tessdata_dir=None):
        """``engines``: a tessapi.Engines (in-process); None reads through the tesseract CLI."""
        self.engines = engines
        self.tessdata_dir = tessdata_dir

    # ------------------------------------------------------------ raw reads

    def image_to_data(self, image, lang, psm, whitelist=None):
        if self.engines is not None:
            return self.engines.image_to_data(image, lang, psm, whitelist)
        config = f"--psm {psm}"
        if self.tessdata_dir:
            config += f" --tessdata-dir {self.tessdata_dir}"
        if whitelist:
            config += f" -c tessedit_char_whitelist={whitelist}"
        return pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)

    def read(self, image, lang, whitelist=None, psm=13):
        """OCR one line image. Confidence is the length-weighted mean word confidence."""
        data = self.image_to_data(image, lang, psm, whitelist)
        words = [(t.strip(), float(c)) for t, c in zip(data["text"], data["conf"]) if t.strip() and float(c) >= 0]
        if not words:
            return Reading("", 0.0)
        total = sum(len(t) for t, _ in words)
        conf = sum(c * len(t) for t, c in words) / total
        return Reading(clean(" ".join(t for t, _ in words)), conf)

    def read_line(self, image, lang):
        """First-pass read: single-line mode reads (and scores) a line far better than
        raw-line mode, but sometimes returns nothing (light-grey Arabic): then raw-line."""
        return self.read(image, lang, psm=7) or self.read(image, lang, psm=13)

    def read_both(self, image):
        return self.read_line(image, "eng"), self.read_line(image, "ara")

    def reread_small_words(self, gray, b, r):
        """Tesseract sizes a line by its tallest glyphs, so smaller words beside big ones
        are misread ("Youtransterred 9 EGP"): read those words again at their own size.
        ``r`` is the line's English reading; returns it with the small words replaced."""
        img = crop(gray, b)
        scale = crop_scale(b)
        data = self.image_to_data(img, "eng", 7)
        words = [(clean(t), float(c), int(left), int(top), int(w), int(h))
                 for t, c, left, top, w, h in zip(data["text"], data["conf"], data["left"], data["top"],
                                                  data["width"], data["height"]) if clean(t) and float(c) >= 0]
        if len(words) < 2:
            return r
        tallest = max(w[5] for w in words)
        out, changed = [], False
        for text, conf, left, top, w, h in words:
            if h < SMALL_WORD * tallest:
                # back to the work image: the crop is padded, framed and scaled
                x0 = b.x - PAD + (left - BORDER) / scale
                y0 = b.y - PAD + (top - BORDER) / scale
                pad = max(2, h / scale / 3)
                box = type(b)(int(x0 - pad), int(y0 - pad), int(w / scale + 2 * pad), int(h / scale + 2 * pad), b.ink)
                again = self.read(crop(gray, box), "eng", psm=7)
                # one word in, one or two words out (small words often run together)
                if again.conf > conf and 0 < len(again.text.split()) <= 2:
                    text, changed = again.text, True
            out.append(text)
        return Reading(" ".join(out), r.conf) if changed else r

    def read_words(self, image, lang, psm=7):
        """Word-level OCR of a line image: (text, conf, left, right) in image pixels."""
        data = self.image_to_data(image, lang, psm)
        words = []
        for t, c, left, width in zip(data["text"], data["conf"], data["left"], data["width"]):
            t = clean(t)
            if t and float(c) >= 0:
                words.append((t, float(c), int(left), int(left + width)))
        return words

    def read_voted(self, gray, b, lang="eng", whitelist=None, normalize=lambda s: s or None,
                   heights=(44, 56, 72), x0=None, x1=None, psm=7, quorum=2):
        """Read a block at several scales and return the majority answer.

        ``normalize`` maps raw OCR text to a canonical value (None when it is not a valid
        value), so votes are cast on clean values. Single-line mode (psm 7) by default:
        raw-line mode tends to invent a leading "1" on digit strings. ``psm`` may be a
        tuple to vote across modes. Stops early once ``quorum`` readings agree.
        """
        psms = psm if isinstance(psm, tuple) else (psm,)
        votes = []
        for th in heights:
            for p in psms:
                r = self.read(crop(gray, b, th, x0, x1), lang, whitelist, p)
                v = normalize(r.text)
                if v:
                    votes.append((v, r.conf))
            if votes and Counter(v for v, _ in votes).most_common(1)[0][1] >= quorum:
                break
        if not votes and 13 not in psms:
            return self.read_voted(gray, b, lang, whitelist, normalize, heights, x0, x1, psm=13, quorum=quorum)
        if not votes:
            return Reading("", 0.0)
        # confidence-weighted vote (+10 so zero-confidence readings still count a little)
        weight = Counter()
        for v, c in votes:
            weight[v] += c + 10
        best = max(weight, key=weight.get)
        agreement = sum(1 for v, _ in votes if v == best) / max(len(votes), quorum)
        return Reading(best, max(c for v, c in votes if v == best) * agreement)

    # ------------------------------------------------------------ mixed-script lines

    def read_mixed(self, gray, b, eng, ara):
        """A line that may mix Arabic and English ("Living Expenses - عيدية").

        The combined ``eng+ara`` model picks a script per word, which a mixed line needs,
        but on a single-script line it sometimes swaps a word for garbage in the other
        script. So its reading is only trusted when it is mixed *and* each script's words
        were also seen by that script's own model; otherwise the better single reading wins.
        """
        combined = self.read_voted(gray, b, "eng+ara", None, lambda t: t or None, heights=(48, 56, 64))
        if has_arabic(combined.text) and has_latin(combined.text) and _confirmed(combined.text, eng, ara):
            return combined
        return pick_script(eng, ara)[0]

    # ------------------------------------------------------------ headline amount

    def read_amount(self, img, gray, b):
        """The headline amount: digits and currency in two colours ("750" black + "EGP"
        orange, or "EGP" dark + "500" teal). The block is split by colour and each part
        read with its own whitelist; None when the block doesn't split into a number and
        a currency. Levels are relative to the block's own paper and darkest ink, so a dim
        or tinted photo splits like a screenshot."""
        region = img[b.y:b.y2, b.x:b.x2]
        g = gray[b.y:b.y2, b.x:b.x2].astype(int)
        chroma = region.max(axis=2).astype(int) - region.min(axis=2)
        paper, darkest = float(np.percentile(g, 90)), float(np.percentile(g, 1))
        span = max(paper - darkest, 1.0)
        ink = g < darkest + 0.8 * span
        tint = float(np.median(chroma[~ink])) if (~ink).any() else 0.0
        colored = np.flatnonzero((ink & (chroma > tint + 0.25 * span)).sum(axis=0) > 1)
        dark = np.flatnonzero((ink & (chroma < tint + 0.15 * span) & (g < darkest + 0.45 * span)).sum(axis=0) > 1)
        if not colored.size or not dark.size:
            return None
        # the two parts sit side by side: left part ends where the right one starts
        if dark.mean() < colored.mean():
            spans = [(dark.min(), min(dark.max() + 1, colored.min())), (colored.min(), colored.max() + 1)]
        else:
            spans = [(colored.min(), min(colored.max() + 1, dark.min())), (dark.min(), dark.max() + 1)]
        parts = []
        for x0, x1 in spans:
            if x1 - x0 < 3:
                return None
            x0, x1 = b.x + int(x0), b.x + int(x1)
            num = self.read_voted(gray, b, "eng", "0123456789,.", fields.amount, x0=x0, x1=x1)
            cur = self.read_voted(gray, b, "eng", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", fields.currency,
                                  heights=(56,), x0=x0, x1=x1, quorum=1)
            parts.append((num, cur))
        (n0, c0), (n1, c1) = parts
        # one part is the number, the other the currency: the pairing read more surely wins
        options = [(n0, c1, False), (n1, c0, True)]
        num, cur, currency_first = max(options, key=lambda o: (bool(o[0]) and bool(o[1]), o[0].conf + o[1].conf))
        if not num or not cur:
            return None
        text = f"{cur.text} {num.text}" if currency_first else f"{num.text} {cur.text}"
        return Reading(text, min(num.conf, cur.conf))

    # ------------------------------------------------------------ masked names

    def read_masked_name(self, gray, b, runs, eng, ara):
        """Names like "ADEL T****** M******" or "محمد ع*** ت****".

        Asterisks confuse both models, so they are found by shape (``runs``) and erased,
        the remaining letters are read, and every asterisk run is re-attached to the
        single initial letter that precedes it in reading order.
        """
        region = gray[b.y:b.y2, b.x:b.x2]
        mask = ink_mask(region)
        erased = gray.copy()
        bg = int(np.median(region[~mask])) if (~mask).any() else 255
        for x0, x1, _ in runs:
            erased[b.y:b.y2, b.x + x0:b.x + x1] = bg
        img = crop(erased, b)
        scale = crop_scale(b)

        # psm 7 splits words far better than raw-line mode once the asterisks are gone
        eng_words, ara_words = self.read_words(img, "eng"), self.read_words(img, "ara")
        if not eng_words and not ara_words:
            eng_words, ara_words = self.read_words(img, "eng", 13), self.read_words(img, "ara", 13)
        eng_r, ara_r = _words_reading(eng_words), _words_reading(ara_words)
        lang = _masked_script(runs, mask.shape[1])
        if lang is None:
            _, lang = pick_script(Reading(eng_r.text, (name_conf(eng_r) + name_conf(eng)) / 2),
                                  Reading(ara_r.text + ara.text, (ara_r.conf + ara.conf) / 2))
        rtl = lang == "ara"

        # tesseract words in region x coordinates: [x0, x1, text, conf]
        words = [[(l - BORDER) / scale - PAD, (r - BORDER) / scale - PAD, t, c]
                 for t, c, l, r in (ara_words if rtl else eng_words)]
        erased_mask = mask.copy()
        for x0, x1, _ in runs:
            erased_mask[:, x0:x1] = False
        star_w = (runs[0][1] - runs[0][0]) / max(runs[0][2], 1)

        items = []  # (x0, x1, text, conf, stars)
        claimed = set()
        for run in runs:
            letter = _initial_letter(erased_mask, run, rtl, star_w * 1.5)
            items.append((run[0], run[1], "", None, run[2]))
            if letter is None:
                continue
            lx0, lx1 = letter
            overlapping = [i for i, w in enumerate(words) if w[0] < lx1 and w[1] > lx0 and i not in claimed]
            word = None
            if overlapping:
                claimed.add(overlapping[0])
                # a word that swallowed the initial ("ADELT", "AO") gives up its last char
                word = (words[overlapping[0]][2][-1], words[overlapping[0]][3])
            # the letter is also read alone: the line read may have dropped it ("Ashraf I****"
            # -> "Ashraf") or joined it to the next initial ("A** M***" -> "AM")
            ch = self._read_letter(erased, b, lx0, lx1, erased_mask, lang, word)
            items.append((lx0, lx1, ch, word[1] if word else None, 0))

        # Masks look like "<full first name(s)> <initial>*** <initial>*** ...": everything
        # before the first masked initial is the unmasked prefix, read on its own with
        # voting (much more reliable than as part of the line); words between masked parts
        # are OCR debris and are dropped.
        letters = [it for it in items if not it[4]]
        if letters:
            first = max(letters, key=lambda it: it[0]) if rtl else min(letters, key=lambda it: it[0])
            px0, px1 = (first[1] + 1, mask.shape[1]) if rtl else (0, first[0] - 1)
            cols = np.flatnonzero(erased_mask[:, max(0, px0):max(0, px1)].any(axis=0))
            if cols.size:
                px0, px1 = px0 + int(cols[0]), px0 + int(cols[-1]) + 1
                script_ok = has_arabic if rtl else has_latin
                prefix = self.read_voted(erased, b, lang, None, lambda t: _name_text(t) if script_ok(t) else None,
                                         heights=(40, 48, 56, 64, 72), x0=b.x + px0, x1=b.x + px1,
                                         psm=(7, 13), quorum=3)
                if not prefix.text:
                    prefix = Reading(" ".join(w[2] for w in words if w[0] >= px0 - 2 and w[1] <= px1 + 2), 0.0)
                if prefix.text:
                    items.append((px0, px1, prefix.text, prefix.conf, 0))
        else:
            items += [(w[0], w[1], w[2], w[3], 0) for w in words if w[2]]
        items.sort(key=lambda it: (it[0] + it[1]) / 2, reverse=rtl)

        out = []
        for _, _, text, _, stars in items:
            if stars:
                if out:
                    out[-1] += "*" * stars
                else:
                    out.append("*" * stars)
            else:
                out.append(text)
        confs = [c for _, _, _, c, stars in items if not stars and c is not None]
        return Reading(" ".join(out), float(np.mean(confs)) if confs else 0.0)

    def _read_letter(self, gray, b, x0, x1, mask, lang, word=None):
        """The initial before an asterisk run (columns x0:x1 of the block), voted across
        sizes and modes; ``word``: (char, conf) the line read gave it, one more vote."""
        # a thin vertical stroke is capital I / alef, which tesseract reads as "|", "o" or nothing
        ys = np.flatnonzero(mask[:, x0:x1].any(axis=1))
        if ys.size and (x1 - x0) <= 0.25 * (ys[-1] - ys[0] + 1):
            return "ا" if lang == "ara" else "I"
        script_ok = has_arabic if lang == "ara" else has_latin
        votes, count = Counter(), Counter()
        if word and script_ok(word[0]):
            votes[word[0]] += word[1] + 10
        for th in LETTER_HEIGHTS:
            for psm in (10, 8):
                r = self.read(crop(gray, b, th, x0=b.x + x0 - LETTER_PAD, x1=b.x + x1 + LETTER_PAD), lang, psm=psm)
                ch = r.text[:1]
                if lang == "ara" and ch and not has_arabic(ch):
                    ch = ARABIC_LOOKALIKE.get(ch, ch)
                if ch and script_ok(ch) and ch not in "|1l!":
                    votes[ch] += r.conf + 10
                    count[ch] += 1
            if count and count.most_common(1)[0][1] >= LETTER_QUORUM:
                break
        return votes.most_common(1)[0][0] if votes else "?"


def _confirmed(text, eng, ara):
    latin = re.findall(r"[A-Za-z]{3,}", text)
    arabic = re.findall("[؀-ۿ]{2,}", text)
    return all(w.lower() in eng.text.lower() for w in latin) and all(w in ara.text for w in arabic)


def asterisk_runs(gray, b):
    """Runs of asterisks in a block as (x0, x1, count), x relative to the block.

    In a small or blurred image neighbouring asterisks touch and form one blob a whole
    number of asterisks wide; such a blob counts as that many.
    """
    mask = ink_mask(gray[b.y:b.y2, b.x:b.x2])
    H = mask.shape[0]
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    # The baseline, from the tall glyphs (capitals, ascenders): their highest bottom.
    # Asterisks float well above it; lower-case letters ("e", "n", "s" in a line whose
    # descenders make it taller) sit on it and are otherwise shaped much like asterisks.
    tall = [y + h for x, y, w, h, _ in stats[1:] if h >= 0.6 * H]
    baseline = min(tall) if tall else H
    stars, wide = [], []
    for x, y, w, h, area in stats[1:]:
        if not (0.18 * H <= h <= 0.6 * H):
            continue
        if (y + h / 2) > 0.62 * H or y + h > baseline - 0.2 * H:  # asterisks float in the upper part
            continue
        # small asterisks render as nearly solid blobs, hence the high upper bound
        if not (0.15 <= area / (w * h) <= 0.85):
            continue
        # a letter dot has its letter underneath; an asterisk has nothing below it
        if mask[min(H, y + h + max(1, int(0.12 * H))):, x:x + w].any():
            continue
        if 0.6 <= w / h <= 1.6:
            stars.append((int(x), int(y), int(w), int(h)))
        elif w / h > 1.6:
            wide.append((int(x), int(y), int(w), int(h)))
    if len(stars) < 2:
        return []
    # asterisks are identical glyphs on one line: keep the candidates of the dominant size
    # and vertical position (drops letter parts)
    my, mw, mh = (float(np.median([s[i] for s in stars])) for i in (1, 2, 3))
    # relative tolerances, but at least a couple of pixels: small asterisks (~10 px) vary
    # by up to 3 px from rendering alone
    tol_h, tol_w = max(0.2 * mh, 3.0), max(0.25 * mw, 3.0)
    same_line = [(x, w) for x, y, w, h in stars
                 if abs(h - mh) <= tol_h and abs(w - mw) <= tol_w and abs(y - my) <= tol_h]
    for x, y, w, h in wide:
        k = round(w / mw)
        if k >= 2 and abs(h - mh) <= tol_h and abs(y - my) <= tol_h and abs(w - k * mw) <= max(0.35 * mw, 3.0):
            same_line += [(round(x + i * w / k), round(w / k)) for i in range(k)]
    runs = []
    for x, w in sorted(same_line):
        if runs and x - runs[-1][1] <= 0.8 * mw + 2:
            runs[-1][1] = x + w
            runs[-1][2] += 1
        else:
            runs.append([x, x + w, 1])
    # a lone "asterisk" is most likely a stray mark; real masks have 2+
    return [tuple(r) for r in runs if r[2] >= 2]


def _initial_letter(mask, run, rtl, star_w):
    """Columns of the single visible letter that precedes an asterisk run."""
    cols = mask.any(axis=0)
    W = cols.size
    step = 1 if rtl else -1
    x = run[1] if rtl else run[0] - 1
    # skip the small gap between the letter and its asterisks
    skipped = 0
    while 0 <= x < W and not cols[x]:
        x += step
        skipped += 1
        if skipped > star_w:
            return None
    start = x
    gap = 0
    while 0 <= x < W and gap <= 1:
        gap = 0 if cols[x] else gap + 1
        x += step
    end = x - step * (gap + 1)
    x0, x1 = sorted((start, end))
    return (x0, x1 + 1) if 0 <= x0 < W else None


def _name_text(text):
    t = re.sub(r"[^\w\s؀-ۿ]", "", text)
    t = re.sub(r"\d", "", t) if has_arabic(t) else t
    return re.sub(r"\s+", " ", t).strip() or None


# Latin / digit look-alikes the Arabic model returns for isolated Arabic letters
ARABIC_LOOKALIKE = {"2": "ع", "3": "ع", "8": "ع", "7": "ح", "0": "ه", "5": "ه", "9": "ة"}


def name_conf(r):
    """English confidence for a *name* line: real names are capitalised, while the English
    model's garbage on Arabic glyphs is mostly lower-case."""
    words = [w for w in r.text.split() if w[:1].isalpha()]
    if not words:
        return r.conf * 0.5
    capitalised = sum(1 for w in words if w[0].isupper()) / len(words)
    return r.conf * (0.4 + 0.6 * capitalised)


def _masked_script(runs, width):
    """The script of a masked name from where its asterisks are: the last part of the name
    is always masked, so the line ends in asterisks on the right for a Latin name and on
    the left for an Arabic one. None when the asterisks at the end weren't found."""
    star_w = (runs[0][1] - runs[0][0]) / max(runs[0][2], 1)
    at_left = runs[0][0] <= 1.5 * star_w
    at_right = width - runs[-1][1] <= 1.5 * star_w
    if at_left != at_right:
        return "ara" if at_left else "eng"
    return None


def _words_reading(words):
    if not words:
        return Reading("", 0.0)
    total = sum(len(t) for t, *_ in words)
    return Reading(" ".join(t for t, *_ in words), sum(c * len(t) for t, c, *_ in words) / total)
