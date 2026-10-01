"""Find text lines on a receipt screenshot with plain OpenCV morphology.

Receipts are app screenshots (or photos of one, whose page image.prepare has already
made even): a light page, anti-aliased text, a few colourful icons. Every run of pixels
darker than the page by a fixed ratio, dilated horizontally, becomes one block (a label,
a value, or a whole line); icons are dropped by colour. Tesseract's own page layout
analysis (psm 11) loses short and light lines that are read fine on their own.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

# Pixels darker than this share of the page's brightness are ink: 215 on a white page,
# where light-grey labels sit around 150-175, the page at 250+, card borders 235-245.
# The page is the PAGE_PERCENTILE of the grey levels, so a page that is not quite white
# (a flattened photo) splits the same way.
INK_RATIO = 215 / 255
PAGE_PERCENTILE = 90
# Width of the window the local page level is taken over (px at ~28 px text): wider than
# the biggest glyph of the headline amount, much narrower than a photo's light fall-off.
LOCAL_PAGE_KERNEL = 160
# Mean chroma (max-min channel) of ink above which a block is an icon or logo; HSV
# saturation is unstable on near-black pixels, absolute chroma is not.
ICON_CHROMA = 45
NEUTRAL_CHROMA = 40
# A colourful block at least this many times as wide as tall may be colour text.
COLOUR_TEXT_ASPECT = 2.0
# Images whose most colourful pixels stay below this chroma are greyscale.
GRAYSCALE_CHROMA = 12
MIN_HEIGHT = 10
MAX_HEIGHT = 160
# A brand wordmark in the header ("axis") is taller than any text line.
MAX_COLOUR_HEIGHT = 320
# An unbroken horizontal run of ink this long is a rule or border (the widest glyph
# stroke, a bold "—" or the bar of a big "7", is far shorter at ~28 px text).
RULE_WIDTH = 200
# Merges the letters and words of a line (text ~28 px tall after image.prepare); the gap
# between a label and its value, or between two columns, is much wider.
DILATE = (28, 3)


@dataclass
class Block:
    x: int
    y: int
    w: int
    h: int
    ink: float  # darkness of the strokes, page = 255; grey labels are lighter than values
    extra: dict = field(default_factory=dict)

    @property
    def x2(self):
        return self.x + self.w

    @property
    def y2(self):
        return self.y + self.h


def page_level(gray):
    return max(1.0, float(np.percentile(gray, PAGE_PERCENTILE)))


def ink_threshold(gray):
    """Grey level below which a pixel is ink, relative to the page."""
    return INK_RATIO * page_level(gray)


def local_page(gray):
    """The page's brightness around each pixel: a max filter wider than any glyph (text
    strokes vanish under it), on a quarter-size copy. Where a photo's page falls off
    towards an edge, ink is still measured against the page right there, not the
    brightest page in the image, so the shaded page doesn't turn into ink."""
    h, w = gray.shape
    small = cv2.resize(gray, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    k = LOCAL_PAGE_KERNEL // 4
    bg = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def find_blocks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    chroma = (img.max(axis=2).astype(np.int16) - img.min(axis=2)).astype(np.uint8)
    page = page_level(gray)
    mask = (gray < INK_RATIO * np.minimum(local_page(gray), page)).astype(np.uint8) * 255
    # Card borders and the shaded edge of a photographed page are no text, but would join
    # every line they touch into one block too big to keep: no glyph stroke runs taller
    # than a text line or wider than RULE_WIDTH unbroken.
    rules = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, MAX_HEIGHT + 1)))
    rules |= cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (RULE_WIDTH, 1)))
    mask &= ~rules

    merged = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, DILATE))
    _, _, stats, _ = cv2.connectedComponentsWithStats(merged)
    half = DILATE[0] // 2
    # undo the dilation's horizontal growth
    pieces = [[int(x) + half, int(y), int(x + w) - half, int(y + h)]
              for x, y, w, h, _ in stats[1:] if h <= MAX_COLOUR_HEIGHT and w > DILATE[0]]
    pieces = _merge_line_pieces(pieces)

    blocks = []
    for x0, y, x1, y1 in pieces:
        x, w, h = x0, x1 - x0, y1 - y
        if not (MIN_HEIGHT <= h <= MAX_COLOUR_HEIGHT) or w < 12:
            continue
        roi_mask = mask[y:y + h, x:x + w] > 0
        if roi_mask.sum() < 15:
            continue
        roi_chroma = chroma[y:y + h, x:x + w][roi_mask]
        ink = min(255.0, float(np.percentile(gray[y:y + h, x:x + w][roi_mask], 5)) * 255 / page)
        # Icons are colour all over; text is mostly neutral ink. "250 EGP" (black digits,
        # orange currency) is half neutral: a plain block.
        if float(roi_chroma.mean()) > ICON_CHROMA and float((roi_chroma < 40).mean()) < 0.35:
            # Neutral text merged into an icon (a small "من" beside a logo) is a block of
            # its own. A wide colour block may be colour text (a wordmark, a teal "EGP 500"):
            # kept as a candidate, the caller decides by how surely it reads.
            blocks += _neutral_blocks(mask, chroma, gray, page, x, y, w, h)
            if w >= COLOUR_TEXT_ASPECT * h:
                blocks.append(Block(int(x), int(y), int(w), int(h), ink, {"colour": True}))
            continue
        if h > MAX_HEIGHT:  # only colour text (a big wordmark) may be this tall
            continue
        blocks.append(Block(int(x), int(y), int(w), int(h), ink))
    if float(np.percentile(chroma, 99.5)) < GRAYSCALE_CHROMA:
        blocks = _drop_gray_icons(blocks)
    blocks.sort(key=lambda b: (b.y, b.x))
    return blocks


def _neutral_blocks(mask, chroma, gray, page, x, y, w, h):
    """Text blocks of neutral ink inside a colourful block's box."""
    neutral = ((mask[y:y + h, x:x + w] > 0) & (chroma[y:y + h, x:x + w] < NEUTRAL_CHROMA)).astype(np.uint8)
    merged = cv2.dilate(neutral, cv2.getStructuringElement(cv2.MORPH_RECT, DILATE))
    _, _, stats, _ = cv2.connectedComponentsWithStats(merged)
    half = DILATE[0] // 2
    out = []
    for sx, sy, sw, sh, _ in stats[1:]:
        sx0, sx1 = sx + half, sx + sw - half
        sw = sx1 - sx0
        if not (MIN_HEIGHT <= sh <= MAX_HEIGHT) or sw < 12 or sw > 0.8 * w:
            continue
        roi = neutral[sy:sy + sh, sx0:sx1] > 0
        if roi.sum() < 15:
            continue
        ink = min(255.0, float(np.percentile(gray[y + sy:y + sy + sh, x + sx0:x + sx1][roi], 5)) * 255 / page)
        out.append(Block(int(x + sx0), int(y + sy), int(sw), int(sh), ink))
    return out


def _drop_gray_icons(blocks):
    """Without colour, icons can't be dropped by chroma; drop them by shape: compact,
    taller than a text line, mid-grey. Text lines are wide; the amount is tall but black."""
    if not blocks:
        return blocks
    line_h = float(np.median([b.h for b in blocks]))
    return [b for b in blocks if not (b.h >= 1.4 * line_h and b.w < 2.2 * b.h and b.ink > 40)]


def _merge_line_pieces(pieces):
    """Join pieces of one big line that the fixed dilation kept apart: in the headline
    amount ("57,475 EGP") the gaps around the comma and before the currency are wider
    than in body text. Label/value gaps are far larger than these."""
    merged = True
    while merged:
        merged = False
        pieces.sort(key=lambda p: p[0])
        for i, a in enumerate(pieces):
            for j in range(i + 1, len(pieces)):
                b = pieces[j]
                ha, hb = a[3] - a[1], b[3] - b[1]
                overlap = min(a[3], b[3]) - max(a[1], b[1])
                gap = b[0] - a[2]
                if overlap >= 0.5 * min(ha, hb) and gap <= 0.45 * max(ha, hb) and max(ha, hb) >= 50:
                    a[:] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del pieces[j]
                    merged = True
                    break
            if merged:
                break
    return pieces
