import json
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from api.persistence import RecordingResponder
from channels.events.base import ChannelEvent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks/whatsapp", tags=["whatsapp"])


@router.get("", response_class=PlainTextResponse)
async def verify_subscription(
    request: Request,
    mode: Optional[str] = Query(None, alias="hub.mode"),
    verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    challenge: Optional[str] = Query(None, alias="hub.challenge"),
) -> str:
    """Meta's one-time handshake when the webhook URL is registered."""
    if not request.app.state.whatsapp.verify_challenge(mode, verify_token):
        raise HTTPException(status_code=403, detail="webhook verification failed")
    return challenge or ""


@router.post("")
async def receive_update(
    request: Request,
    background_tasks: BackgroundTasks,
    signature: Optional[str] = Header(None, alias="X-Hub-Signature-256"),
) -> dict:
    """Verify, parse and route a WhatsApp update.

    Meta retries deliveries that are not acknowledged quickly, so the reply
    (media download, OCR, send) runs after the 200 has been returned.
    """
    channel = request.app.state.whatsapp
    body = await request.body()
    if not channel.verify(body, signature):
        logger.warning("whatsapp webhook rejected: bad signature")
        raise HTTPException(status_code=403, detail="invalid signature")

    try:
        payload = json.loads(body)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="unexpected payload")

    event = channel.parse(payload)
    if event is not None:
        logger.info("whatsapp event %s from %s", type(event).__name__, event.conversation_id)
        background_tasks.add_task(_handle, request.app.state.responder, event)
    return {"status": "ok"}


async def _handle(responder: RecordingResponder, event: ChannelEvent) -> None:
    try:
        await responder.handle(event)
    except Exception:
        logger.exception("failed to handle whatsapp event %s", event.provider_message_id)
