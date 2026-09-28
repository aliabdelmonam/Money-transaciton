"""Debug overlay: token boxes coloured by the role the parser assigned."""

import cv2
import numpy as np

from ..extraction.layout import Token
from .image import save_image

_ROLE_COLORS = {"label": (140, 140, 140), "value": (0, 140, 255), "sentence": (200, 0, 200)}


def role_color(role: str) -> tuple[int, int, int]:
    if role.startswith("anchor"):
        return 0, 0, 230
    if role.startswith("sender"):
        return 230, 120, 0
    if role.startswith("receiver"):
        return 0, 160, 0
    return _ROLE_COLORS.get(role, (80, 80, 80))


def draw_debug(img: np.ndarray, tokens: list[Token], path: str) -> None:
    vis = img.copy()
    for t in tokens:
        c = role_color(t.role)
        cv2.rectangle(vis, (t.x1, t.y1), (t.x2, t.y2), c, 2)
        if t.role:
            cv2.putText(vis, t.role, (t.x1, max(12, t.y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    save_image(path, vis)
