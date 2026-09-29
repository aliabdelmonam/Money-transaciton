import hashlib
import hmac
import logging
from typing import Optional

from channels.events.base import ChannelEvent
from channels.services.mapper import MessageMapper

logger = logging.getLogger(__name__)


class WebhookService:
    def __init__(
        self,
        mapper: MessageMapper,
        secret: Optional[str] = None,
        scheme: str = "hmac_sha256",
    ):
        self._mapper = mapper
        self._secret = secret
        self._scheme = scheme

    def verify(self, body: bytes, signature: Optional[str]) -> bool:
        if not self._secret:
            logger.error("webhook rejected: no secret configured")
            return False
        if self._scheme == "secret_token":
            return hmac.compare_digest((signature or "").encode(), self._secret.encode())
        expected = hmac.new(self._secret.encode(), body, hashlib.sha256).hexdigest()
        provided = (signature or "").split("=")[-1]
        # Compare bytes: compare_digest raises on non-ASCII str, e.g. a garbage header.
        return hmac.compare_digest(provided.encode(), expected.encode())

    def parse(self, payload: dict) -> Optional[ChannelEvent]:
        return self._mapper.to_event(payload)