"""HTTP entry point: WhatsApp webhook -> receipt OCR -> text reply.

Run from the project root (so ``paddle_ocr`` / ``transaction_extractor`` import):

    python -m uvicorn api.main:app --app-dir src --host 0.0.0.0 --port 8000
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

import channels.adapters.whatsapp.adapter  # noqa: F401  registers the "whatsapp" channel
from api.persistence import RecordingResponder
from api.receipts import ReceiptReader, TransactionReplyProvider
from api.routes import health, whatsapp
from api.security import require_api_key
from channels.builders.channel import ChannelBuilder
from channels.config import Settings
from channels.events.messages import ImageMessageReceived
from channels.registry import channel_registry
from channels.services import ChatbotResponder, FallbackReplyProvider, StaticReplyProvider
from db import Database
from db.migrate import upgrade as upgrade_database
from db.transactions import TransactionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

FALLBACK_REPLY = (
    "Sorry, I couldn't read the transaction details from this image. "
    "Please send a clear screenshot of the receipt."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.api_key = settings.api_key.get_secret_value() if settings.api_key else None
    if not app.state.api_key:
        logger.warning("API_KEY is not set: every API endpoint will reject every request")
    if settings.database_migrate_on_startup:
        await asyncio.to_thread(upgrade_database, settings.database_url)
    database = Database(settings.database_url)
    store = TransactionStore(database)
    channel = ChannelBuilder(settings.channels).build("whatsapp", channel_registry.get("whatsapp"))

    providers = []
    reader = None
    if settings.ocr_enabled:
        reader = ReceiptReader(settings.inbound_media_dir, settings.ocr_workers)
        logger.info("loading OCR models...")
        await reader.start()
        providers.append(TransactionReplyProvider(channel, reader, store))
    providers.append(StaticReplyProvider({ImageMessageReceived: FALLBACK_REPLY}))

    app.state.whatsapp = channel
    app.state.responder = RecordingResponder(
        ChatbotResponder(channel, FallbackReplyProvider(*providers)), store
    )
    try:
        yield
    finally:
        await channel.close()
        if reader is not None:
            reader.close()
        await database.dispose()


def create_app() -> FastAPI:
    application = FastAPI(title="Money Transaction Bot", lifespan=lifespan)
    # Every router needs the X-API-Key header, except the webhook: Meta cannot send
    # it, so those calls are authenticated by their signature and verify token instead.
    protected = [Depends(require_api_key)]
    application.include_router(health.router, dependencies=protected)
    application.include_router(whatsapp.router)
    return application


app = create_app()
