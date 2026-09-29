import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

from api.persistence import record
from channels.events.base import ChannelEvent
from channels.events.messages import ImageMessageReceived
from channels.protocol.channel import Channel
from db.transactions import KEY_FIELDS, TransactionStore

logger = logging.getLogger(__name__)

SUMMARY_FIELDS = (
    ("provider", "Provider"),
    ("status", "Status"),
    ("type", "Type"),
    ("amount", "Amount"),
    ("fees", "Fees"),
    ("total", "Total"),
    ("reference", "Reference"),
    ("datetime", "Date"),
    ("note", "Note"),
)
MONEY_FIELDS = ("amount", "fees", "total")


class ReceiptReader:
    """Run the PaddleOCR transaction extractor on a single worker thread.

    PaddleOCR is CPU-heavy and not thread-safe, so every call goes through
    one dedicated thread and the event loop is never blocked.
    """

    def __init__(self, media_dir: Path):
        self._media_dir = Path(media_dir)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")
        self._ocr = None

    async def start(self) -> None:
        """Load the OCR models up front so the first receipt is not slowed down."""
        await self._run(self._load)

    async def read(self, data: bytes, filename: str) -> dict:
        """Save the image to the inbound media dir and extract its transaction fields."""
        return await self._run(self._extract, self._media_dir / Path(filename).name, data)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def _run(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    def _load(self) -> None:
        from paddle_ocr import create_ocr  # heavy import, only when OCR is enabled

        self._ocr = create_ocr()
        # paddle_ocr disables logging process-wide on import; give the app its logs back.
        logging.disable(logging.NOTSET)
        logger.info("OCR models loaded")

    def _extract(self, path: Path, data: bytes) -> dict:
        from paddle_ocr import extract_lines
        from transaction_extractor import extract

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        result, _ = extract(extract_lines(self._ocr, path))
        return result


class TransactionReplyProvider:
    """Reply to a receipt image with the transaction details read from it, and store them."""

    def __init__(
        self, channel: Channel, reader: ReceiptReader, store: TransactionStore, max_concurrent: int = 2
    ):
        self._channel = channel
        self._reader = reader
        self._store = store
        # OCR itself is serial (one worker thread); this bounds how many images are
        # downloaded and held in memory while waiting for it.
        self._slots = asyncio.Semaphore(max_concurrent)

    async def reply(self, event: ChannelEvent) -> Optional[str]:
        if not isinstance(event, ImageMessageReceived):
            return None
        async with self._slots:
            media = await self._channel.fetch_media(event.attachment)
            if media is None:
                logger.warning("could not download media for message %s", event.provider_message_id)
                await record("transaction", self._store.record_failure(event, "could not download media"))
                return None
            try:
                result = await self._reader.read(media.data, media.filename or "receipt")
            except Exception as error:
                await record("transaction", self._store.record_failure(event, repr(error), media))
                raise
        logger.info("extracted transaction from %s: %s", event.provider_message_id, result)
        await record("transaction", self._store.record_result(event, media, result))
        return format_transaction(result)


def format_transaction(result: dict) -> Optional[str]:
    """Render an extractor result as a WhatsApp message; ``None`` if nothing was found."""
    if all(result.get(key) is None for key in KEY_FIELDS):
        return None

    lines = ["*Transaction details*"]
    currency = result.get("currency")
    for key, label in SUMMARY_FIELDS:
        value = result.get(key)
        if value is None:
            continue
        if key in MONEY_FIELDS and currency:
            value = f"{value} {currency}"
        elif key == "datetime":
            value = format_date(value)
        lines.append(f"{label}: {value}")
    for party, label in (("sender", "From"), ("receiver", "To")):
        details = {slot: value for slot, value in (result.get(party) or {}).items() if slot != "other"}
        if details:
            lines.append(f"{label}: " + ", ".join(str(value) for value in details.values()))
    return "\n".join(lines)


def format_date(value: str) -> str:
    """ISO date from the extractor -> '24 Sep 2026, 10:11 PM' ('24 Sep 2026' if it has no time)."""
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return value
    text = f"{dt.day} {dt:%b %Y}"
    if "T" in value:
        text += f", {dt:%I:%M %p}".replace(" 0", " ", 1)
    return text
