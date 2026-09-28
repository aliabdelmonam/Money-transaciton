"""Image I/O (unicode-safe on Windows) and OCR preprocessing."""

import os

import cv2
import numpy as np

from src.config.ai import ImageConfig


def load_image(path: str) -> np.ndarray:
    return decode_image(np.fromfile(path, dtype=np.uint8), source=path)


def decode_image(data: bytes | np.ndarray, source: str = "<bytes>") -> np.ndarray:
    """Any image bytes -> BGR uint8 (transparent PNGs flattened on white)."""
    buf = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray)) else data
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read image: {source}")
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / max(1, img.max()))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3].astype(np.float32) * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
    return img


def save_image(path: str, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if ok:
        buf.tofile(path)


def resize_for_ocr(img: np.ndarray, cfg: ImageConfig) -> tuple[np.ndarray, float]:
    """Upscale small screenshots / downscale huge photos. Returns (image, scale)."""
    long_side = max(img.shape[:2])
    s = 1.0
    if long_side < cfg.target_long_side:
        s = cfg.target_long_side / long_side
    elif long_side > cfg.max_long_side:
        s = cfg.max_long_side / long_side
    if s != 1.0:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    return img, s
