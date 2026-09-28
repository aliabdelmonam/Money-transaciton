"""Provider detection, label matching and from/to anchor matching."""

from dataclasses import dataclass
from typing import Optional

from ..text.normalize import clean_text, norm, ratio
from ..knowledge import ANCHOR_WORDS, GENERIC_LABELS, PROVIDERS
from .layout import Token


@dataclass
class LabelMatch:
    field: str
    score: float
    remainder: list[str]     # words on the token that are not part of the label (inline value)


@dataclass
class Anchor:
    side: str                # sender | receiver
    line: int
    token: Token
    after: list[str]         # 'To Mobile Wallet' -> ['Mobile', 'Wallet']
    before: list[str]


def detect_provider(tokens: list[Token]) -> str:
    blob = " ".join(norm(a) for t in tokens for a in t.alts)
    best, best_score = "generic", 0
    for name, prof in PROVIDERS.items():
        score = sum(1 for kw in prof["detect"] if norm(kw) and norm(kw) in blob)
        if score > best_score:
            best, best_score = name, score
    return best


def build_label_map(provider: str) -> dict[str, str]:
    """normalized label -> field"""
    lm = {norm(lab): fld for fld, labels in GENERIC_LABELS.items() for lab in labels}
    for lab, fld in PROVIDERS.get(provider, {}).get("label_overrides", {}).items():
        lm[norm(lab)] = fld
    return lm


class LabelMatcher:
    def __init__(self, label_map: dict[str, str], fuzzy_threshold: float):
        self.label_map = label_map
        self.fuzzy_threshold = fuzzy_threshold

    def _score(self, words: list[str], full: str, lab: str):
        if full == lab:
            return 100.0, []
        nl = len(lab.split())
        if len(words) > nl:
            if norm(" ".join(words[:nl])) == lab:
                return 96.0, words[nl:]
            if norm(" ".join(words[-nl:])) == lab:
                return 95.0, words[:-nl]
        if len(lab) >= 5:
            r = ratio(full, lab)
            if r >= self.fuzzy_threshold:
                return r, []
        return None

    def match(self, tok: Token) -> Optional[LabelMatch]:
        best = None     # (field, score, remainder, label_len)
        for alt in tok.alts:
            words = clean_text(alt).split()
            full = norm(alt)
            if not words or not full:
                continue
            for lab, fld in self.label_map.items():
                cand = self._score(words, full, lab)
                if cand and (best is None or (cand[0], len(lab)) > (best[1], best[3])):
                    best = (fld, cand[0], cand[1], len(lab))
        return LabelMatch(best[0], best[1], best[2]) if best else None


def match_anchor(tok: Token) -> Optional[tuple[str, list[str], list[str]]]:
    """'From' / 'To Mobile Wallet' / 'إلى انستاباي' / 'من' -> (side, words_after, words_before)"""
    for alt in tok.alts:
        words = clean_text(alt).split()
        if not words or len(words) > 5:
            continue
        nw = [norm(w) for w in words]
        for side, keys in ANCHOR_WORDS.items():
            for i in {0, len(nw) - 1}:
                if nw[i] in keys:
                    return side, words[i + 1:], words[:i]
    return None
