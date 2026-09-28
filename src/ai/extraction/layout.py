"""Tokens (OCR results + parse state) grouped into visual lines."""

from dataclasses import dataclass
from statistics import mean, median

from ..protocols.ocr import OCRResult
from ..text.normalize import script_counts


@dataclass(eq=False)
class Token:
    """An OCR result plus the role the parser assigned to it."""
    ocr: OCRResult
    role: str = ""
    line: int = -1

    @property
    def text(self) -> str: return self.ocr.text
    @property
    def alts(self) -> tuple[str, ...]: return self.ocr.readings
    @property
    def x1(self) -> int: return self.ocr.bbox.x1
    @property
    def y1(self) -> int: return self.ocr.bbox.y1
    @property
    def x2(self) -> int: return self.ocr.bbox.x2
    @property
    def y2(self) -> int: return self.ocr.bbox.y2
    @property
    def cx(self) -> float: return self.ocr.bbox.cx
    @property
    def cy(self) -> float: return self.ocr.bbox.cy
    @property
    def h(self) -> int: return self.ocr.bbox.height


def join_tokens(toks: list[Token]) -> str:
    if not toks:
        return ""
    ar, la = script_counts(" ".join(t.text for t in toks))
    order = sorted(toks, key=lambda t: t.x1, reverse=ar > la)   # RTL for Arabic lines
    return " ".join(t.text for t in order).strip()


class Line:
    def __init__(self, idx: int, tokens: list[Token]):
        self.idx = idx
        self.tokens = sorted(tokens, key=lambda t: t.x1)

    @property
    def cy(self) -> float:
        return float(mean(t.cy for t in self.tokens))

    @property
    def h(self) -> float:
        return float(median(t.h for t in self.tokens))

    def text(self, toks: list[Token] | None = None) -> str:
        return join_tokens(self.tokens if toks is None else toks)

    def free(self) -> list[Token]:
        return [t for t in self.tokens if not t.role]


def group_lines(tokens: list[Token], y_tolerance: float) -> list[Line]:
    rows: list[list[Token]] = []
    for t in sorted(tokens, key=lambda t: t.cy):
        for row in reversed(rows[-3:]):
            rcy = mean(x.cy for x in row)
            rh = median(x.h for x in row)
            if abs(t.cy - rcy) <= y_tolerance * min(t.h, rh):
                row.append(t)
                break
        else:
            rows.append([t])
    lines = [Line(i, r) for i, r in enumerate(rows)]
    for ln in lines:
        for t in ln.tokens:
            t.line = ln.idx
    return lines
