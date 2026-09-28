"""Structured output of receipt extraction."""

from dataclasses import asdict, dataclass, field, fields
from typing import Optional

MONEY_FIELDS = ("amount", "fees", "total")


@dataclass
class Party:
    name: Optional[str] = None
    name_alt: Optional[str] = None      # e.g. Arabic + English name of the same person
    phone: Optional[str] = None
    email: Optional[str] = None
    bank: Optional[str] = None
    account_type: Optional[str] = None

    @classmethod
    def keys(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))


@dataclass
class Receipt:
    provider: str = "generic"
    status: Optional[str] = None        # success | failed | None
    sender: Party = field(default_factory=Party)
    receiver: Party = field(default_factory=Party)
    amount: Optional[float] = None
    fees: Optional[float] = None
    total: Optional[float] = None
    currency: Optional[str] = None
    reference: Optional[str] = None
    date: Optional[str] = None          # raw text as read
    date_iso: Optional[str] = None
    note: Optional[str] = None
    other_phones: list[str] = field(default_factory=list)
    derived: list[str] = field(default_factory=list)    # fields computed, not read
    warnings: list[str] = field(default_factory=list)

    def party(self, side: str) -> Party:
        return self.sender if side == "sender" else self.receiver

    def to_dict(self) -> dict:
        return asdict(self)
