"""Image preparation before line finding: whatever the upload, dark text on an even white
page with text lines about TEXT_HEIGHT px tall, so every pixel threshold downstream holds.

  - a light receipt on a dark surround is cut out, and straightened when the screen was
    photographed at an angle; a dark-mode screenshot is inverted and its page stretched
    to white
  - a photo of a screen (darker towards the corners, tinted, blurred) has its lighting
    flattened and is sharpened; a light-grey or tinted screenshot page is scaled to white;
    a white page is left exactly as it is
  - wide empty margins (a screenshot pasted on a big canvas) are cut away
  - the image is scaled by its text height, not its width, so a cropped, a padded, a small
    and a huge screenshot all end up the same
  - a tilt of up to MAX_SKEW degrees is measured and rotated away

``prepare`` also returns how to map a box back to the uploaded image's pixels.
"""
import cv2
import numpy as np

from . import segment

# Images are first probed at this width, then scaled so that text lines are about
# TEXT_HEIGHT px tall; within TEXT_HEIGHT_TOLERANCE they are left at that size.
NORMALIZED_WIDTH = 1000
TEXT_HEIGHT = 28
TEXT_HEIGHT_TOLERANCE = 0.15
MAX_WORK_PIXELS = 12_000_000
# Skew beyond this is not a tilted screenshot; below the minimum, leave it be.
MAX_SKEW, MIN_SKEW = 10.0, 0.3
# Empty margins are cut away when the content fills less than this share of the width
# or height; a pixel is content when it differs this much from the border's median.
MIN_MARGIN_CROP = 0.5
CONTENT_CONTRAST = 25
# The page behind the text is estimated on a copy this wide with a closing kernel of
# BG_KERNEL px: much wider than a text stroke, much narrower than lighting changes.
BG_WIDTH = 250
BG_KERNEL = 15
# A page at least this bright all over (90% of an 8x8 grid of cells) is a screenshot and
# is left as it is; screen photos measure 200-215.
FLAT_PAGE = 245
# So is a page of one even level: when UNIFORM_SHARE of the cells lie within
# +-UNIFORM_LEVELS of one level (screenshots 0.5-1, photos 0.1-0.35). Such a page is only
# scaled to white, and only when every channel is at least UNIFORM_MIN_PAGE; white cards
# on a grey page (at most PAGE_REGION_LEVELS brighter than it) keep their pixels.
UNIFORM_LEVELS = 4
UNIFORM_SHARE = 0.5
UNIFORM_MIN_PAGE = 200
PAGE_REGION_LEVELS = 8
# Large dark areas (an icon, the desk) are not page: the page estimate never drops below
# this share of the brightest page, so they are brightened at most 2x.
MIN_PAGE_SHARE = 0.5
# Unsharp mask for photos: blurred light-grey labels otherwise break into specks.
SHARPEN_SIGMA = 3.0
SHARPEN_AMOUNT = 1.0
# A page cut out of a photo loses 1/PAGE_INSET of its size on each side (its blurred edge).
PAGE_INSET = 100
# An inverted dark-mode page (#121212 -> 237) is stretched to white; the page is this
# percentile of the grey levels.
STRETCH_PERCENTILE = 90


def load(path):
    """BGR image from a file (any path, unlike cv2.imread on Windows); transparency on white."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not decode image {path}")
    if img.dtype != np.uint8:  # 16-bit PNG
        img = (img / 257).round().astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        # IMREAD_COLOR would expose whatever colour transparent pixels hold, often black
        alpha = img[..., 3:].astype(np.float32) / 255
        img = (img[..., :3].astype(np.float32) * alpha + 255 * (1 - alpha)).round().astype(np.uint8)
    return img


def prepare(img):
    """(work image, (x0, y0, scale)): a work-image point (x, y) is (x0 + x / scale,
    y0 + y / scale) in ``img``, give or take the deskew / perspective correction."""
    work, offset, _ = prepare_with_blocks(img)
    return work, offset


def prepare_with_blocks(img):
    """``prepare``, plus the work image's segment.find_blocks: measuring the text height
    already found them, so a caller that needs them doesn't pay for them twice."""
    img, (ox, oy) = _undark(img)
    img = _flatten(img)
    img, (mx, my) = _crop_margins(img)
    ox, oy = ox + mx, oy + my
    scale = NORMALIZED_WIDTH / img.shape[1]
    # A probe at a fixed width measures the text; when it is off, measure again at the
    # corrected scale (very small text is measured badly).
    for _ in range(3):
        probe = _resize(img, scale)
        probe = _rotate(probe, _skew_angle(probe))
        blocks = segment.find_blocks(probe)
        height = _text_height(blocks)
        if height is None or abs(TEXT_HEIGHT / height - 1) <= TEXT_HEIGHT_TOLERANCE:
            break
        new_scale = _clamp_scale(img, scale * TEXT_HEIGHT / height)
        if abs(new_scale / scale - 1) <= TEXT_HEIGHT_TOLERANCE:
            break
        scale = new_scale
    return probe, (ox, oy, scale), blocks


def _border(gray):
    h, w = gray.shape
    m = max(2, min(h, w) // 50)
    return np.concatenate([gray[:m].ravel(), gray[-m:].ravel(), gray[:, :m].ravel(), gray[:, -m:].ravel()])


def _crop_margins(img):
    """Cut away wide empty margins; an image already filled by its content is unchanged."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    content = np.abs(gray.astype(np.int16) - int(np.median(_border(gray)))) > CONTENT_CONTRAST
    # rows / columns with more than a few content pixels: JPEG specks don't count
    rows = np.flatnonzero(content.sum(axis=1) > max(2, w // 500))
    cols = np.flatnonzero(content.sum(axis=0) > max(2, h // 500))
    if not rows.size or not cols.size:
        return img, (0, 0)
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    if (x1 - x0) > MIN_MARGIN_CROP * w and (y1 - y0) > MIN_MARGIN_CROP * h:
        return img, (0, 0)
    pad = max(8, (x1 - x0) // 50)
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    return img[y0:min(h, y1 + pad), x0:min(w, x1 + pad)], (int(x0), int(y0))


def _text_height(blocks):
    """Typical text line height: the lower quartile of block heights (mostly labels and
    values rather than the big amount or icons)."""
    if len(blocks) < 3:
        return None
    return float(np.percentile([b.h for b in blocks], 25))


def _clamp_scale(img, scale):
    h, w = img.shape[:2]
    return min(scale, (MAX_WORK_PIXELS / (h * w)) ** 0.5)


def _resize(img, scale):
    if abs(scale - 1) < 1e-3:
        return img
    h, w = img.shape[:2]
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interp)


def _undark(img):
    """Dark text on a light page: a light receipt on a dark surround is cut out, an image
    dark all over (dark mode) is inverted. Returns the image and its offset in ``img``."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if np.median(_border(gray)) >= 100:
        return img, (0, 0)
    # light = brighter than Otsu's split between surround and page: a photographed page
    # may be far from white, dimmer still towards the corners
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    light = cv2.morphologyEx((gray > otsu).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(light)
    if n >= 2:
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[i, cv2.CC_STAT_AREA] >= 0.2 * h * w:
            return _cut_out(img, labels == i, stats[i])
    # no light page inside the dark frame: the whole screen is dark mode
    return (_stretch(255 - img) if np.median(gray) < 100 else img), (0, 0)


def _cut_out(img, page, stats):
    """The light page without its dark surround: an upright rectangle is cropped, a
    slanted four-sided page (a photo taken at an angle) warped back to a rectangle, any
    other shape keeps its bounding box with the surround painted white."""
    x, y, bw, bh = (int(v) for v in stats[:4])
    h, w = page.shape
    contours, _ = cv2.findContours(page.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    # the corners are the hull's extreme points towards each corner of the image
    pts = hull.reshape(-1, 2).astype(np.float32)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    tl, tr, br, bl = pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]
    src = np.float32([tl, tr, br, bl])
    corners = np.float32([[x, y], [x + bw - 1, y], [x + bw - 1, y + bh - 1], [x, y + bh - 1]])
    if cv2.contourArea(src) >= 0.9 * cv2.contourArea(hull):
        if np.abs(src - corners).max() <= 1:
            out = img[y:y + bh, x:x + bw]
        else:
            pw = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
            ph = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
            dst = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]])
            out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (pw, ph),
                                      flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    else:
        inside = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(inside, hull, 1)
        out = img.copy()
        out[inside == 0] = 255
        out = out[y:y + bh, x:x + bw]
    inset = max(2, min(out.shape[:2]) // PAGE_INSET)
    return out[inset:-inset, inset:-inset], (x + inset, y + inset)


def _stretch(img):
    """Scale an inverted dark-mode screen so its page (dark grey, not black, so light grey
    once inverted) is white."""
    page = float(np.percentile(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), STRETCH_PERCENTILE))
    if page >= 254 or page < 128:
        return img
    return np.clip(img.astype(np.float32) * (255.0 / page), 0, 255).round().astype(np.uint8)


def _flatten(img):
    """Make the page evenly white. A photo of a screen is divided, channel by channel, by
    an estimate of the page behind the text (a morphological close that removes the
    strokes, on a small copy) and sharpened. A screenshot, whose page is one even level,
    is not reshaped: a white one is left as it is, a grey or tinted one scaled to white."""
    h, w = img.shape[:2]
    f = min(1.0, BG_WIDTH / w)
    small = cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BG_KERNEL, BG_KERNEL))
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    # the brightest page in each cell of an 8x8 grid (icons and panels don't fill a cell)
    page = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    sh, sw = page.shape
    cells = np.array([page[r * sh // 8:(r + 1) * sh // 8, c * sw // 8:(c + 1) * sw // 8].max()
                      for r in range(8) for c in range(8) if sh >= 8 and sw >= 8], np.int16)
    if not cells.size or np.percentile(cells, 10) >= FLAT_PAGE:
        return img
    # the share of cells at the most common page level: is there any fall-off?
    if max(np.mean(np.abs(cells - v) <= UNIFORM_LEVELS) for v in np.unique(cells)) >= UNIFORM_SHARE:
        return _whiten(img, small, page)
    bg = np.maximum(bg, MIN_PAGE_SHARE * np.percentile(bg, 95, axis=(0, 1)))
    bg = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), BG_KERNEL / 4)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    out = img.astype(np.float32) * (255.0 / np.maximum(bg, 1.0))
    out = cv2.addWeighted(out, 1 + SHARPEN_AMOUNT, cv2.GaussianBlur(out, (0, 0), SHARPEN_SIGMA), -SHARPEN_AMOUNT, 0)
    return np.clip(out, 0, 255).round().astype(np.uint8)


def _whiten(img, small, page):
    """Scale an evenly lit grey / tinted page to white, one gain per channel, only where
    the page is at that level: white cards on it keep their pixels (scaling them too thins
    the anti-aliased edges of their text)."""
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    level = int(np.argmax(np.bincount(gray.ravel(), minlength=256)))
    colour = np.median(small[np.abs(gray.astype(np.int16) - level) <= UNIFORM_LEVELS], axis=0)
    if colour.min() < UNIFORM_MIN_PAGE or colour.min() >= FLAT_PAGE:
        return img
    gain = (255.0 / colour).astype(np.float32)
    on_page = (page.astype(np.int16) <= level + PAGE_REGION_LEVELS).astype(np.float32)
    weight = cv2.resize(on_page, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)[..., None]
    out = img.astype(np.float32) * (1.0 + weight * (gain - 1.0))
    return np.clip(out, 0, 255).round().astype(np.uint8)


def _skew_angle(img):
    """Tilt of a screenshot, 0 when there is none to correct: the angle at which the ink's
    row profile is sharpest (text rows and the gaps between them line up)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    ink = (small < segment.ink_threshold(small)).astype(np.float32)
    h, w = ink.shape

    def sharpness(angle):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return float(np.var(np.diff(cv2.warpAffine(ink, m, (w, h)).sum(axis=1))))

    coarse = max(np.arange(-MAX_SKEW, MAX_SKEW + 0.01, 0.5), key=sharpness)
    angle = max(np.arange(coarse - 0.5, coarse + 0.51, 0.1), key=sharpness)
    return 0.0 if abs(angle) < MIN_SKEW else float(angle)


def _rotate(img, angle):
    if not angle:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
