import logging
from channels.models.exceptions import ChannelError
from typing import Dict,Type,Callable


logger = logging.getLogger(__name__)



class ChannelRegistry:
    def __init__(self):
        self._adapters:Dict[str,Type] = {}

    def register(self,name: str)-> Callable[[Type],Type]:
        """Class decorator that maps a channel name to its adapter class."""

        def decorator(adapter_cls:Type) -> Type:
            existing = self._adapters.get(name)
            if existing is not None and existing is not adapter_cls:
                raise ChannelError(f"Channel already registered: {name}")

            self._adapters[name]=adapter_cls
            logger.debug("registered channel: %s -> %s", name, adapter_cls.__name__)
            return adapter_cls

        return decorator

    def get(self,name: str) -> Type:
        adapter_cls = self._adapters.get(name)

        if adapter_cls is None:
            raise ChannelError(f"Unknown channel: {name}. Available: {self.names()}")
        return adapter_cls

    def names(self)-> list[str]:
        return sorted(self._adapters)

channel_registry = ChannelRegistry()
register_channel = channel_registry.register