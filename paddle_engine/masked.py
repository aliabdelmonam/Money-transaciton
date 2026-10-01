"""Masked names: "AHMED H****** Y****", "محمد ع*** ت**** م***".

A recognizer can't count repeated characters reliably (CTC merges them, and runs of
asterisks lose or gain one), the detector splits a name at its asterisks, and the Arabic
model reads a lone initial as a Latin look-alike. So, as in the Tesseract engine, masked
lines are found by the shape of their asterisks, and rebuilt: the asterisks are counted,
erased, and the unmasked prefix and each initial are read on their own, the initials with a
letters-only whitelist and voted across paddings.
"""
import re
from collections import Counter

import numpy as np

from tesseract_engine import segment
from tesseract_engine.reader import ARABIC_LOOKALIKE, _initial_letter, _masked_script, asterisk_runs, ink_mask

from . import boxes
from .recognize import AR_LETTERS, ARABIC, LA_LETTERS, LA_UPPER, clean, fix_text

# Each initial is read with this much padding (x the line height), the votes summed.
LETTER_PADS = (0.25, 0.45, 0.7)
PREFIX_PAD = 0.3
# Same-row blocks this close (x the line height) are parts of one masked name.
PART_GAP = 1.5
# A letter this narrow for its height is a thin stroke: I / alef.
THIN_STROKE = 0.25


def read_masked_lines(rec_ar, rec_en, img, gray, blocks, lines, min_conf):
    """Replace the lines that hold a masked name by one rebuilt line.

    ``lines``: dicts with "box" and the "ar" / "en" (text, conf) readings, in work-image
    pixels; a rebuilt line gets "final": (text, conf). ``blocks``: segment.find_blocks of
    ``img``. Masked lines are found among those: the morphology line finder's blocks are
    tight to the ink of the whole line, which is what asterisk_runs measures against (a
    detector box may hold only the asterisks).
    """
    blocks = [[b.x, b.y, b.x2, b.y2] for b in blocks]
    starred = [b for b in blocks if asterisk_runs(gray, _block(b))]
    lines = list(lines)
    done = []
    for box in starred:
        if any(boxes.intersects(box, d) for d in done):
            continue
        box = _whole_name(box, blocks, starred, lines)
        done.append(box)
        group = sorted((i for i, l in enumerate(lines) if l and boxes.intersects(l["box"], box)
                        and boxes.row_overlap(l["box"], box) >= 0.5), key=lambda i: lines[i]["box"][0])
        readings = {k: " ".join(lines[i][k][0] for i in group) for k in ("ar", "en")}
        rebuilt = _read(rec_ar, rec_en, img, gray, box, readings["ar"])
        if rebuilt is None:
            continue
        # asterisks blurred into one blob are not found by shape: when both line readings
        # see more masked parts than were found, the line reading is the better one
        parts = lambda t: len(re.findall(r"\*+", t))
        if group and parts(rebuilt[0]) < min(parts(readings["ar"]), parts(readings["en"])):
            continue
        # the asterisks were found by shape: the structure vouches for the reading
        final = (fix_text(rebuilt[0]), max(rebuilt[1], min_conf))
        if group:
            lines[group[0]] = {"box": box, "ar": lines[group[0]]["ar"], "en": lines[group[0]]["en"], "final": final}
            for i in group[1:]:
                lines[i] = None
        else:  # a light-grey name the detector missed
            lines.append({"box": box, "ar": ("", 0.0), "en": ("", 0.0), "final": final})
    return [l for l in lines if l]


def _block(b):
    return segment.Block(b[0], b[1], b[2] - b[0], b[3] - b[1], 0.0)


def _whole_name(box, blocks, starred, lines):
    """A starred block joined with the same-row blocks of the same name that the line
    finder's dilation kept apart (another starred block, or one the same detection spans)."""
    h = boxes.height(box)
    for b in blocks:
        if b is box or boxes.row_overlap(b, box) < 0.5 or boxes.gap(b, box) > PART_GAP * h \
                or not 0.5 <= boxes.height(b) / h <= 2:
            continue
        if b in starred or any(l and boxes.intersects(b, l["box"]) and boxes.intersects(box, l["box"]) for l in lines):
            box = boxes.union(box, b)
    return box


def _token(text, rtl, single):
    """A recognized name part in its script; ``single``: an initial (one letter)."""
    if rtl:
        text = clean(text)
        if ARABIC.search(text):
            text = re.sub(r"[^ء-يٱ-ۓ\s]", "", text)
        else:
            text = ARABIC_LOOKALIKE.get(text[:1], "") if single else ""
    else:
        # "|" / "!" are a capital I here, not edge noise
        text = re.sub(r"[^A-Za-z\s]", "", re.sub(r"[|!]", "I", text))
        if single:
            text = text[:1].upper()
    return " ".join(text.split())


def _read(rec_ar, rec_en, img, gray, b, line_text):
    """(text, conf) of the masked name in box ``b``; None when it has no asterisk runs."""
    runs = asterisk_runs(gray, _block(b))
    if not runs:
        return None
    region = gray[b[1]:b[3], b[0]:b[2]]
    mask = ink_mask(region)
    width = region.shape[1]
    h = boxes.height(b)
    # the last part of a name is always masked: asterisks at the left end -> Arabic
    rtl = (_masked_script(runs, width) or ("ara" if ARABIC.search(line_text) else "eng")) == "ara"
    rec = rec_ar if rtl or rec_en is None else rec_en

    erased_mask, work = mask.copy(), img.copy()
    page = np.median(img[b[1]:b[3], b[0]:b[2]][~mask], axis=0) if (~mask).any() else 255
    for x0, x1, _ in runs:
        erased_mask[:, x0:x1] = False
        work[b[1]:b[3], b[0] + x0:b[0] + x1] = page
    star_w = (runs[0][1] - runs[0][0]) / max(runs[0][2], 1)
    letters = [_initial_letter(erased_mask, run, rtl, star_w * 1.5) for run in runs]

    items = []  # (x centre, text, asterisks, conf)
    # the unmasked prefix: everything before the first initial in reading order
    found = [l for l in letters if l]
    if found:
        first = max(found) if rtl else min(found)
        px0, px1 = (first[1] + 1, width) if rtl else (0, first[0] - 1)
    else:
        px0, px1 = (runs[-1][1], width) if rtl else (0, runs[0][0])
    cols = np.flatnonzero(erased_mask[:, max(0, px0):max(0, px1)].any(axis=0))
    if cols.size:
        px0, px1 = px0 + int(cols[0]), px0 + int(cols[-1]) + 1
        crop = boxes.crop(work, [b[0] + px0, b[1], b[0] + px1, b[3]], max(3, int(PREFIX_PAD * h)))
        (text, conf), = rec.read([crop], (AR_LETTERS if rtl else LA_LETTERS) + " ")
        if text := _token(text, rtl, False):
            items.append(((px0 + px1) / 2, text, 0, conf))

    # each initial alone (its neighbours blanked out), voted across paddings
    jobs = []
    for k, letter in enumerate(letters):
        if not letter:
            continue
        lx0, lx1 = letter
        ys = np.flatnonzero(erased_mask[:, lx0:lx1].any(axis=1))
        if ys.size and lx1 - lx0 <= THIN_STROKE * (ys[-1] - ys[0] + 1):
            items.append(((lx0 + lx1) / 2, "ا" if rtl else "I", 0, None))
            continue
        alone = work.copy()
        alone[b[1]:b[3], b[0]:b[0] + lx0] = page
        alone[b[1]:b[3], b[0] + lx1:b[2]] = page
        for pad in LETTER_PADS:
            jobs.append((k, boxes.crop(alone, [b[0] + lx0, b[1], b[0] + lx1, b[3]], max(3, int(pad * h)))))
    votes = {}
    for (k, _), (text, conf) in zip(jobs, rec.read([c for _, c in jobs], AR_LETTERS if rtl else LA_UPPER)):
        if text := _token(text, rtl, True):
            votes.setdefault(k, Counter())[text] += conf
    for k, v in votes.items():
        text, score = v.most_common(1)[0]
        items.append((sum(letters[k]) / 2, text, 0, score / len(LETTER_PADS)))

    items += [((x0 + x1) / 2, "", n, None) for x0, x1, n in runs]
    items.sort(key=lambda it: it[0], reverse=rtl)
    out = ""
    for _, text, stars, _ in items:
        out += "*" * stars if stars else (" " if out else "") + text
    confs = [c for *_, c in items if c is not None]
    return out.strip(), float(np.mean(confs)) if confs else 0.0
