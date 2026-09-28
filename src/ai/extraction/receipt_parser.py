"""Layout-aware receipt parser.

Steps, each claiming the tokens it uses (via `Token.role`) so later steps skip them:
  1. sentence patterns ("You transferred 9 EGP to 010...")
  2. from / to anchors
  3. "label : value" on the same line, or the line above / below
  4. sender / receiver blocks under their anchors
  5. simple fields, headline amount, status / currency
  6. amount + fees = total cross-check, missing ones derived
"""

import re
from dataclasses import dataclass, field

from src.config.ai import ExtractionConfig
from ..protocols.ocr import OCRResult
from ..text.normalize import clean_text, norm, norm_soft
from ..text.parsers import (clean_name, find_bank, find_email, find_phone, is_name, letter_count,
                            parse_datetime, parse_money, parse_reference)
from ..knowledge import CURRENCY_MARKERS, FAIL_WORDS, GENERIC_SENTENCES, PROVIDERS, SUCCESS_WORDS
from .layout import Line, Token, group_lines, join_tokens
from .matching import Anchor, LabelMatcher, build_label_map, detect_provider, match_anchor
from .models import MONEY_FIELDS, Party, Receipt


@dataclass
class KeyValue:
    value: str
    line: int
    tokens: list[Token] = field(default_factory=list)

    def candidates(self) -> list[str]:
        return [self.value] + [a for t in self.tokens for a in t.alts]


class LayoutReceiptExtractor:
    """Implements `ReceiptExtractor`. Stateless: safe to reuse across images."""

    def __init__(self, config: ExtractionConfig | None = None):
        self.config = config or ExtractionConfig()

    def extract(self, results: list[OCRResult]) -> Receipt:
        return self.parse(results)[0]

    def parse(self, results: list[OCRResult]) -> tuple[Receipt, list[Line], list[Token]]:
        """Also returns lines / tokens (with roles) for debugging and text dumps."""
        tokens = [Token(r) for r in results]
        lines = group_lines(tokens, self.config.line_y_tolerance)
        receipt = _ParseRun(lines, tokens, self.config).run()
        return receipt, lines, tokens


class _ParseRun:
    """State of parsing one receipt."""

    def __init__(self, lines: list[Line], tokens: list[Token], config: ExtractionConfig):
        self.lines = lines
        self.tokens = tokens
        self.cfg = config
        self.label_lines: set[int] = set()
        provider = detect_provider(tokens)
        self.labels = LabelMatcher(build_label_map(provider), config.fuzzy_label_threshold)
        self.r = Receipt(provider=provider)

    def run(self) -> Receipt:
        self._sentences()
        anchors = self._anchors()
        kv = self._key_values(anchors)
        self._parties(anchors, kv)
        self._simple_fields(kv)
        self._headline_amount()
        self._status_currency()
        self._validate_money()
        self._other_phones()
        return self.r

    # ---- 1. sentence receipts ---------------------------------------------------
    def _sentences(self):
        pats = GENERIC_SENTENCES + PROVIDERS.get(self.r.provider, {}).get("sentences", [])
        for ln in self.lines:
            free = ln.free()
            if not free:
                continue
            cands = [ln.text(free)] + [a for t in free for a in t.alts]
            m = next((m for pat in pats for c in cands if (m := re.search(pat, norm_soft(c)))), None)
            if not m:
                continue
            g = m.groupdict()
            if g.get("amount") and self.r.amount is None:
                self.r.amount = parse_money([g["amount"]])
            for side in ("sender", "receiver"):
                party = self.r.party(side)
                if g.get(side + "_phone") and not party.phone:
                    party.phone = find_phone([g[side + "_phone"]])
            for t in free:
                t.role = "sentence"

    # ---- 2. from / to anchors ---------------------------------------------------
    def _anchors(self) -> list[Anchor]:
        found = []
        for ln in self.lines:
            for t in ln.free():
                a = match_anchor(t)
                if a:
                    side, after, before = a
                    t.role = "anchor_" + side
                    found.append(Anchor(side, ln.idx, t, after, before))
        return found

    # ---- 3. label : value -------------------------------------------------------
    @staticmethod
    def _value_ok(fld: str, value: str) -> bool:
        if not value:
            return False
        if fld in MONEY_FIELDS:
            return parse_money([value]) is not None
        if fld == "date":
            return parse_datetime(value) is not None
        if fld == "reference":
            return parse_reference([value]) is not None
        if fld.endswith(".phone"):
            return find_phone([value]) is not None
        return letter_count(value) >= 2

    def _neighbour_value(self, ln: Line, fld: str, anchor_lines: set[int]):
        order = (-1, 1) if fld in MONEY_FIELDS else (1, -1)   # headline amount sits above its label
        for d in order:
            j = ln.idx + d
            if not 0 <= j < len(self.lines) or j in anchor_lines:
                continue
            nb = self.lines[j]
            if abs(nb.cy - ln.cy) > 3.0 * max(ln.h, nb.h):
                continue
            free = nb.free()
            if not free or any(self.labels.match(t) for t in free):
                continue
            val = nb.text(free)
            if self._value_ok(fld, val):
                return val, free
        return None

    def _key_values(self, anchors: list[Anchor]) -> dict[str, list[KeyValue]]:
        kv: dict[str, list[KeyValue]] = {}
        anchor_lines = {a.line for a in anchors}
        for ln in self.lines:
            hits = [(t, m) for t in ln.free() if (m := self.labels.match(t))]
            if not hits:
                continue
            self.label_lines.add(ln.idx)
            for t, _ in hits:
                t.role = "label"
            others = ln.free()
            for t, m in hits:
                if len(hits) > 1:   # two columns on one line -> nearest label owns the value
                    mine = [o for o in others if min(hits, key=lambda h: abs(h[0].cx - o.cx))[0] is t]
                else:
                    mine = others
                value = " ".join(m.remainder + ([ln.text(mine)] if mine else [])).strip()
                vtoks = list(mine)
                if not self._value_ok(m.field, value):
                    nb = self._neighbour_value(ln, m.field, anchor_lines)
                    if nb:
                        value, vtoks = nb
                for v in vtoks:
                    v.role = "value"
                if value:
                    kv.setdefault(m.field, []).append(KeyValue(value, ln.idx, vtoks))
        return kv

    # ---- 4. sender / receiver blocks --------------------------------------------
    def _read_party(self, toks: list[Token], anchor: Anchor | None, side: str) -> Party:
        info = Party()
        extra: list[str] = []
        if anchor:
            if anchor.after:
                text = " ".join(anchor.after)
                if find_phone([text]):
                    info.phone = find_phone([text])
                elif find_email([text]):
                    info.email = find_email([text])
                else:
                    info.account_type = text
            extra += anchor.before
        name_lines: dict[int, list[Token]] = {}
        at = anchor.token if anchor else None
        for t in toks:
            texts = [t.text, *t.alts]
            if em := find_email(texts):
                info.email = info.email or em
                t.role = side + "_email"
                continue
            if ph := find_phone(texts):
                info.phone = info.phone or ph
                t.role = side + "_phone"
                continue
            if bank := find_bank(texts):
                info.bank = info.bank or bank
                t.role = side + "_bank"
                continue
            # bank logos / wallet icons ('BANQUE MISR', 'WLS') sit beside the text column,
            # completely outside the From/To anchor's x-range -> not the name
            if at and t.line != at.line and (t.x2 < at.x1 or t.x1 > at.x2):
                continue
            if is_name(clean_name(t.text)):
                name_lines.setdefault(t.line, []).append(t)
        info.bank = info.bank or find_bank(extra)
        names: list[str] = []
        for li in sorted(name_lines):
            nm = clean_name(join_tokens(name_lines[li]))
            if is_name(nm) and nm not in names:
                names.append(nm)
            for t in name_lines[li]:
                t.role = side + "_name"
        if names:
            info.name = names[0]
        if len(names) > 1:
            info.name_alt = names[1]
        return info

    def _block_below(self, a: Anchor, stop: set[int]) -> list[int]:
        idxs = [a.line]
        for j in range(a.line + 1, min(len(self.lines), a.line + 1 + self.cfg.max_party_lines)):
            if j in stop:
                break
            cur, prev = self.lines[j], self.lines[j - 1]
            if cur.cy - prev.cy > self.cfg.party_gap_ratio * max(cur.h, prev.h):
                break
            idxs.append(j)
        return idxs

    def _block_above(self, start: int, stop: set[int]) -> list[int]:
        idxs = []
        for j in range(start - 1, max(-1, start - 1 - self.cfg.max_party_lines), -1):
            if j in stop or any(t.role == "sentence" for t in self.lines[j].tokens):
                break
            idxs.insert(0, j)
        return idxs

    def _parties(self, anchors: list[Anchor], kv: dict[str, list[KeyValue]]):
        first: dict[str, Anchor] = {}
        for a in sorted(anchors, key=lambda a: a.line):
            first.setdefault(a.side, a)
        stop = set(self.label_lines) | {a.line for a in first.values()}
        blocks: dict[str, tuple[list[int], Anchor | None]] = {
            side: (self._block_below(a, stop), a) for side, a in first.items()}
        # only 'to/إلى' found -> the block right above it is the sender
        if "receiver" in blocks and "sender" not in blocks:
            idxs = self._block_above(blocks["receiver"][1].line, stop)
            if idxs:
                blocks["sender"] = (idxs, None)
                self.r.warnings.append("sender block inferred from position (no 'From/من' found)")
        for side, (idxs, a) in blocks.items():
            toks = [t for j in idxs for t in self.lines[j].tokens if not t.role]
            info = self._read_party(toks, a, side)
            party = self.r.party(side)
            for k in Party.keys():
                if getattr(info, k) and not getattr(party, k):
                    setattr(party, k, getattr(info, k))
            for t in toks:
                t.role = t.role or side
        # explicit labels (اسم المستقبل / رقم المحفظة ...) win over block inference
        for fld, items in kv.items():
            if "." not in fld:
                continue
            side, key = fld.split(".")
            val = items[0].value
            if key == "phone":
                val = find_phone(items[0].candidates()) or val
            elif key == "name":
                val = clean_name(val) or val
            setattr(self.r.party(side), key, val)

    # ---- 5. simple fields -------------------------------------------------------
    def _simple_fields(self, kv: dict[str, list[KeyValue]]):
        amounts = kv.get("amount", [])
        if len(amounts) > 1 and "total" not in kv and "fees" in kv:
            later = [a for a in amounts[1:] if a.line > kv["fees"][0].line]
            if later:
                kv["total"] = [later[0]]
                amounts.remove(later[0])
                self.r.warnings.append("second 'amount' label below fees treated as total")
        for fld in MONEY_FIELDS:
            if getattr(self.r, fld) is not None:
                continue
            for it in kv.get(fld, []):
                v = parse_money(it.candidates())
                if v is not None:
                    setattr(self.r, fld, v)
                    break
        for it in kv.get("reference", []):
            if ref := parse_reference(it.candidates()):
                self.r.reference = ref
                break
        for it in kv.get("date", []):
            for c in it.candidates():
                if iso := parse_datetime(c):
                    self.r.date, self.r.date_iso = clean_text(c), iso
                    break
            if self.r.date_iso:
                break
        if not self.r.date_iso:                # no date label -> scan free text
            for ln in self.lines:
                txt = ln.text([t for t in ln.tokens if t.role != "label"])
                if iso := parse_datetime(txt):
                    self.r.date, self.r.date_iso = txt, iso
                    break
        if kv.get("note"):
            self.r.note = clean_text(kv["note"][0].value)

    def _headline_amount(self):
        if self.r.amount is not None:
            return
        best = None
        for t in self.tokens:
            if t.role in ("label", "value"):
                continue
            n = " " + norm(" ".join(t.alts)) + " "
            if not any(c in n for c in CURRENCY_MARKERS):
                continue
            v = parse_money(t.alts)
            if v is not None and (best is None or t.h > best[0]):
                best = (t.h, v, t)
        if best:
            self.r.amount = best[1]
            best[2].role = best[2].role or "value"
            self.r.warnings.append("amount taken from the biggest EGP number (no label)")

    def _status_currency(self):
        blob = " " + " ".join(norm(a) for t in self.tokens for a in t.alts) + " "
        if any(norm(w) in blob for w in FAIL_WORDS):
            self.r.status = "failed"
        elif any(norm(w) in blob for w in SUCCESS_WORDS):
            self.r.status = "success"
        if any(c in blob for c in CURRENCY_MARKERS[:3]) or self.r.amount is not None:
            self.r.currency = "EGP"

    # ---- 6. money cross-check ---------------------------------------------------
    def _validate_money(self):
        r = self.r
        a, f, t = r.amount, r.fees, r.total
        if a is not None and f is not None and t is not None:
            if abs(a + f - t) > 0.01:
                if abs(a - t) <= 0.01:
                    r.warnings.append(f"fees read as {f} but amount == total -> fees set to 0")
                    r.fees = 0.0
                else:
                    r.warnings.append("amount + fees != total, please check this receipt")
        elif a is not None and f is not None:
            r.total = round(a + f, 2)
            r.derived.append("total")
        elif a is not None and t is not None and t >= a:
            r.fees = round(t - a, 2)
            r.derived.append("fees")
        elif f is not None and t is not None:
            r.amount = round(t - f, 2)
            r.derived.append("amount")

    def _other_phones(self):
        assigned = {self.r.sender.phone, self.r.receiver.phone}
        for t in self.tokens:
            if t.role in ("label", "value") or t.role.startswith("anchor"):
                continue
            ph = find_phone([t.text])
            if ph and ph not in assigned and ph not in self.r.other_phones:
                self.r.other_phones.append(ph)
