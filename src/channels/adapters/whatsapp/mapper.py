import logging
from datetime import datetime, timezone
from typing import Iterator, Optional

from channels.events.base import ChannelEvent
from channels.events.messages import ImageMessageReceived, MessageDelivered, MessageRead
from channels.models.attachment import Attachment, AttachmentType
from channels.models.user import User

logger = logging.getLogger(__name__)

CHANNEL = "whatsapp"


class WhatsAppMessageMapper:
    """Translate WhatsApp Cloud API webhook payloads into channel events.

    Only image messages are mapped; every other message type (text,
    audio, video, document, location, ...) is ignored and yields ``None``.
    """

    def to_event(self, payload: dict) -> Optional[ChannelEvent]:
        for value in self._values(payload):
            contacts = value.get("contacts") or []
            for message in value.get("messages") or []:
                event = self._message_event(message, contacts)
                if event is not None:
                    return event
            for status in value.get("statuses") or []:
                event = self._status_event(status)
                if event is not None:
                    return event
        return None

    @staticmethod
    def inbound_phone_number_id(payload: dict) -> Optional[str]:
        for value in WhatsAppMessageMapper._values(payload):
            phone_number_id = (value.get("metadata") or {}).get("phone_number_id")
            if phone_number_id:
                return phone_number_id
        return None

    @staticmethod
    def _values(payload: dict) -> Iterator[dict]:
        if payload.get("object") != "whatsapp_business_account":
            return
        for entry in payload.get("entry") or []:
            for change in entry.get("changes") or []:
                if change.get("field") != "messages":
                    continue
                value = change.get("value")
                if isinstance(value, dict):
                    yield value

    def _message_event(
        self, message: dict, contacts: list[dict]
    ) -> Optional[ChannelEvent]:
        message_type = message.get("type")
        if message_type != "image":
            logger.info("whatsapp: ignoring non-image message type=%s", message_type)
            return None

        image = message.get("image") or {}
        media_id = image.get("id")
        sender_id = message.get("from")
        if not media_id or not sender_id:
            logger.warning("whatsapp: image message missing id/from: %s", message)
            return None

        attachment = Attachment(
            type=AttachmentType.IMAGE,
            ref=media_id,
            caption=image.get("caption"),
            mime_type=image.get("mime_type"),
            filename=f"{media_id}{self._extension(image.get('mime_type'))}",
            metadata={
                "sha256": image.get("sha256"),
                "timestamp": self._timestamp(message.get("timestamp")),
            },
        )
        return ImageMessageReceived(
            channel=CHANNEL,
            conversation_id=sender_id,
            provider_message_id=message.get("id"),
            sender=User(id=sender_id, name=self._contact_name(contacts, sender_id)),
            attachment=attachment,
        )

    @staticmethod
    def _status_event(status: dict) -> Optional[ChannelEvent]:
        message_id = status.get("id")
        recipient = status.get("recipient_id")
        if not message_id or not recipient:
            return None
        kind = status.get("status")
        if kind == "delivered":
            return MessageDelivered(
                channel=CHANNEL, conversation_id=recipient, message_id=message_id
            )
        if kind == "read":
            return MessageRead(
                channel=CHANNEL, conversation_id=recipient, message_id=message_id
            )
        return None

    @staticmethod
    def _contact_name(contacts: list[dict], wa_id: str) -> Optional[str]:
        for contact in contacts:
            if contact.get("wa_id") == wa_id:
                return (contact.get("profile") or {}).get("name")
        return None

    @staticmethod
    def _timestamp(raw: Optional[str]) -> Optional[str]:
        try:
            return datetime.fromtimestamp(int(raw), tz=timezone.utc).isoformat()
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extension(mime_type: Optional[str]) -> str:
        return {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }.get(mime_type or "", "")
