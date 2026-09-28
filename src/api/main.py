"""HTTP entry point: WhatsApp webhook -> receipt OCR -> text reply.

Run from the project root (so ``paddle_ocr`` / ``transaction_extractor`` import):

    python -m uvicorn api.main:app --app-dir src --host 0.0.0.0 --port 8000
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

import channels.adapters.whatsapp.adapter  # noqa: F401  registers the "whatsapp" channel
from api.receipts import ReceiptReader, TransactionReplyProvider
from api.routes import health, whatsapp
from channels.builders.channel import ChannelBuilder
from channels.config import Settings
from channels.events.messages import ImageMessageReceived
from channels.registry import channel_registry
from channels.services import ChatbotResponder, FallbackReplyProvider, StaticReplyProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

FALLBACK_REPLY = (
    "Sorry, I couldn't read the transaction details from this image. "
    "Please send a clear screenshot of the receipt."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    channel = ChannelBuilder(settings.channels).build("whatsapp", channel_registry.get("whatsapp"))

    providers = []
    reader = None
    if settings.ocr_enabled:
        reader = ReceiptReader(settings.inbound_media_dir)
        logger.info("loading OCR models...")
        await reader.start()
        providers.append(TransactionReplyProvider(channel, reader))
    providers.append(StaticReplyProvider({ImageMessageReceived: FALLBACK_REPLY}))

    app.state.whatsapp = channel
    app.state.responder = ChatbotResponder(channel, FallbackReplyProvider(*providers))
    try:
        yield
    finally:
        await channel.close()
        if reader is not None:
            reader.close()


app = FastAPI(title="Money Transaction Bot", lifespan=lifespan)
app.include_router(health.router)
app.include_router(whatsapp.router)
