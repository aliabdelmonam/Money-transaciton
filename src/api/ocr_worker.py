"""Code that runs inside each OCR worker process.

Kept apart from ``api.receipts`` so a worker only imports the OCR engine and the
extractor, not the web app. Every function here must be importable by name:
Windows starts workers with ``spawn``, which pickles functions by reference.
"""
import importlib
import logging
import signal
import time
from pathlib import Path

# OCR_ENGINE -> root-level module with create_ocr / detect_device / extract_lines
ENGINES = {"paddle": "paddle_ocr", "tesseract": "tesseract_ocr"}

_engine = None
_ocr = None
_device = None


def load(engine: str, device: str, cpu_threads: int) -> None:
    """Process initializer: load the OCR models once per worker.

    ``device`` is ``auto`` (the GPU if there is one), ``cpu`` or e.g. ``gpu:0``.
    """
    global _engine, _ocr, _device
    # Ctrl+C reaches the whole console; the parent shuts the workers down itself.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _engine = importlib.import_module(ENGINES[engine])  # heavy import, only when OCR is enabled

    _device = _engine.detect_device() if device == "auto" else device
    _ocr = _engine.create_ocr(cpu_threads=cpu_threads, device=_device)
    # paddle_ocr disables logging process-wide on import; give the process its logs back.
    logging.disable(logging.NOTSET)


def ping() -> str:
    """Task used to wait until a worker is up; returns the device its models run on."""
    return _device


def read(path: Path, data: bytes) -> tuple[dict, dict]:
    """Save the image and extract its transaction fields; also return stage timings."""
    from transaction_extractor import extract

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    start = time.perf_counter()
    lines = _engine.extract_lines(_ocr, path)
    ocr_done = time.perf_counter()
    result, _ = extract(lines)
    timings = {"ocr": ocr_done - start, "extract": time.perf_counter() - ocr_done, "lines": len(lines)}
    return result, timings
