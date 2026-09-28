import logging
from typing import Type

import httpx

from channels.protocol.channel import Channel
from channels.services.webhook import WebhookService

logger = logging.getLogger(__name__)


class ChannelBuilder:
    def __init__(self, settings):
        self._settings = settings

    def build(self, name: str, adapter_cls: Type) -> Channel:
        config = self._settings.config_for(name)
        client = httpx.AsyncClient(base_url=adapter_cls.base_url(config), timeout=15.0)
        mapper = adapter_cls.mapper_class()
        webhook = WebhookService(
            mapper,
            secret=getattr(config, "webhook_secret", None),
            scheme=getattr(adapter_cls, "webhook_scheme", "hmac_sha256"),
        )
        dependencies = {"webhook": webhook, "client": client, "config": config}
        sender_cls = getattr(adapter_cls, "sender_class", None)
        if sender_cls is not None:
            dependencies["sender"] = sender_cls(client, mapper)
        adapter = adapter_cls(**dependencies)
        logger.info("built channel adapter: %s", name)
        return adapter