from enum import Enum
from typing import Iterator


class Capability(str, Enum):
    IMAGE = "image"

class Capabilities:
    def __init__(self, *capabilities: Capability):
        self._supported = frozenset(capabilities)

    def supports(self, capability: Capability) -> bool:
        return capability in self._supported

    def __contains__(self, capability: Capability) -> bool:
        return capability in self._supported

    def __iter__(self) -> Iterator[Capability]:
        return iter(self._supported)

    def __repr__(self) -> str:
        return f"Capabilities({', '.join(c.value for c in self._supported)})"