"""In-process Tesseract through its C API (libtesseract), called with ctypes.

pytesseract starts a tesseract.exe per call and every process loads the models again,
which costs more than reading a line. Here each model is loaded once per engine and
the engines are reused, about 4x faster for the ~50 line reads of one receipt. Results
match the CLI's: the image goes in as 8-bit grey at the 70 dpi the CLI assumes for a PNG
without one, and the TSV output is parsed the way pytesseract parses it.

``Engines.create`` returns None when the library can't be loaded (or ``OCR_BACKEND=cli``);
the caller then uses pytesseract. A crash inside the library takes the whole process
down, where a crashing tesseract.exe only fails one call.
"""
import atexit
import ctypes
import ctypes.util
import os
import shutil
import sys
import threading
from contextlib import contextmanager

import numpy as np
import pytesseract

# The CLI's resolution for an image that doesn't state one (a PNG from pytesseract).
DPI = 70
TSV_COLUMNS = ["level", "page_num", "block_num", "par_num", "line_num", "word_num",
               "left", "top", "width", "height", "conf", "text"]

_lib = None
_lib_lock = threading.Lock()
_lib_loaded = False
# Engines share a process-wide model cache: create and destroy them one at a time.
_init_lock = threading.Lock()
_all_engines = []


def _library_path():
    path = os.environ.get("TESSERACT_LIB")
    if path:
        return path
    if sys.platform == "win32":
        exe = shutil.which(pytesseract.pytesseract.tesseract_cmd)
        if exe:
            dll = os.path.join(os.path.dirname(os.path.abspath(exe)), "libtesseract-5.dll")
            if os.path.exists(dll):
                return dll
        return None
    return ctypes.util.find_library("tesseract")


def _load():
    path = _library_path()
    if not path:
        return None
    if sys.platform == "win32":
        # its dependencies live beside it; never pick them up from PATH
        os.add_dll_directory(os.path.dirname(os.path.abspath(path)))
    lib = ctypes.CDLL(path)
    h = ctypes.c_void_p
    for name, args, res in [
        ("TessVersion", [], ctypes.c_char_p),
        ("TessBaseAPICreate", [], h),
        ("TessBaseAPIInit3", [h, ctypes.c_char_p, ctypes.c_char_p], ctypes.c_int),
        ("TessBaseAPISetPageSegMode", [h, ctypes.c_int], None),
        ("TessBaseAPISetVariable", [h, ctypes.c_char_p, ctypes.c_char_p], ctypes.c_int),
        ("TessBaseAPISetImage", [h, h, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int], None),
        ("TessBaseAPISetSourceResolution", [h, ctypes.c_int], None),
        ("TessBaseAPIGetTsvText", [h, ctypes.c_int], h),
        ("TessDeleteText", [h], None),
        ("TessBaseAPIClear", [h], None),
        ("TessBaseAPIEnd", [h], None),
        ("TessBaseAPIDelete", [h], None),
    ]:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, res
    version = lib.TessVersion().decode().lstrip("v")  # "5.5.3.20260724"
    return lib if int(version.split(".")[0]) >= 5 else None


def library():
    """The loaded libtesseract, or None (missing, too old, or ``OCR_BACKEND=cli``)."""
    global _lib, _lib_loaded
    with _lib_lock:
        if not _lib_loaded:
            _lib_loaded = True
            if os.environ.get("OCR_BACKEND", "").lower() != "cli":
                try:
                    _lib = _load()
                except (OSError, AttributeError, ValueError):
                    _lib = None
    return _lib


class Engines:
    """Pools of loaded Tesseract engines, one pool per language string, at most
    ``workers`` recognitions at a time. An engine is held for one call only."""

    def __init__(self, lib, tessdata_dir, workers):
        self.lib = lib
        self.tessdata_dir = tessdata_dir
        self.slots = threading.BoundedSemaphore(workers)
        self.idle = {}
        self.idle_lock = threading.Lock()

    @classmethod
    def create(cls, tessdata_dir, workers):
        lib = library()
        return cls(lib, tessdata_dir, workers) if lib else None

    def _new_engine(self, lang):
        datapath = self.tessdata_dir
        # tesseract opens files with narrow fopen: the ANSI code page on Windows
        encoded = datapath.encode("mbcs" if sys.platform == "win32" else sys.getfilesystemencoding()) if datapath else None
        with _init_lock:
            engine = self.lib.TessBaseAPICreate()
            if self.lib.TessBaseAPIInit3(engine, encoded, lang.encode()) != 0:
                self.lib.TessBaseAPIDelete(engine)
                raise RuntimeError(f"tesseract could not load language {lang!r} from {datapath or 'default tessdata'}")
            if not _all_engines:
                # a crop with no ink makes tesseract print debug statistics ("Total count=0")
                # to stderr; debug_file is process-wide, so set it once
                self.lib.TessBaseAPISetVariable(engine, b"debug_file", os.devnull.encode())
            _all_engines.append((self.lib, engine))
        return engine

    @contextmanager
    def _engine(self, lang):
        with self.slots:
            with self.idle_lock:
                pool = self.idle.setdefault(lang, [])
                engine = pool.pop() if pool else None
            if engine is None:
                engine = self._new_engine(lang)
            try:
                yield engine
            finally:
                with self.idle_lock:
                    self.idle[lang].append(engine)

    def image_to_data(self, image, lang, psm, whitelist=None):
        """Same result as ``pytesseract.image_to_data(..., output_type=Output.DICT)``."""
        if image.dtype != np.uint8 or image.ndim != 2:
            raise ValueError("expected a 2-D uint8 grey image")
        image = np.ascontiguousarray(image)
        lib = self.lib
        with self._engine(lang) as engine:
            try:
                lib.TessBaseAPISetPageSegMode(engine, psm)
                if whitelist:
                    lib.TessBaseAPISetVariable(engine, b"tessedit_char_whitelist", whitelist.encode())
                lib.TessBaseAPISetImage(engine, image.ctypes.data, image.shape[1], image.shape[0], 1, image.strides[0])
                lib.TessBaseAPISetSourceResolution(engine, DPI)
                ptr = lib.TessBaseAPIGetTsvText(engine, 0)
                if not ptr:
                    raise RuntimeError("tesseract failed to recognise the image")
                try:
                    tsv = ctypes.string_at(ptr).decode("utf-8")
                finally:
                    lib.TessDeleteText(ptr)
            finally:
                if whitelist:
                    lib.TessBaseAPISetVariable(engine, b"tessedit_char_whitelist", b"")
                lib.TessBaseAPIClear(engine)
        return parse_tsv(tsv)


def parse_tsv(tsv):
    # pytesseract.file_to_dict: numbers truncated with int(float()), text as is
    result = {c: [] for c in TSV_COLUMNS}
    text_col = len(TSV_COLUMNS) - 1
    for line in tsv.strip("\n").split("\n"):
        cells = line.split("\t")
        if len(cells) < text_col:
            continue
        cells += [""] * (len(TSV_COLUMNS) - len(cells))
        for i, col in enumerate(TSV_COLUMNS):
            result[col].append(cells[i] if i == text_col else int(float(cells[i])))
    return result


@atexit.register
def _shutdown():
    # unfreed engines make tesseract print "LEAK" warnings at exit
    with _init_lock:
        for lib, engine in _all_engines:
            lib.TessBaseAPIEnd(engine)
            lib.TessBaseAPIDelete(engine)
        _all_engines.clear()
