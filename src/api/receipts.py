import asyncio
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from pathlib import Path
from typing import Optional

from api import ocr_worker
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
    """Run the PaddleOCR transaction extractor in a pool of worker processes.

    Each worker loads its own copy of the models and reads one image at a time,
    so ``workers`` receipts are read in parallel. Processes rather than threads:
    PaddleOCR is not thread-safe, and running it on a thread stalls the event
    loop for the whole read, which makes every in-flight Meta call time out.
    """

    def __init__(self, media_dir: Path, workers: int = 2):
        self._media_dir = Path(media_dir)
        self._workers = workers
        # Split the cores between workers instead of each one using Paddle's default.
        self._cpu_threads = max(1, (os.cpu_count() or 1) // workers)
        self._executor = self._new_executor()

    async def start(self) -> None:
        """Start every worker and load its models up front so the first receipts are not slowed down."""
        devices = await asyncio.gather(*(self._run(ocr_worker.ping) for _ in range(self._workers)))
        logger.info("OCR models loaded in %d workers on %s (%d CPU threads each)",
                    self._workers, ", ".join(sorted(set(devices))), self._cpu_threads)

    async def read(self, data: bytes, filename: str) -> dict:
        """Save the image to the inbound media dir and extract its transaction fields."""
        result, _ = await self.read_timed(data, filename)
        return result

    async def read_timed(self, data: bytes, filename: str) -> tuple[dict, dict]:
        """``read``, plus how long OCR and extraction took inside the worker."""
        path = self._media_dir / Path(filename).name
        result, timings = await self._run(ocr_worker.read, path, data)
        logger.info("read %s: ocr %.1fs, extract %.2fs", path.name, timings["ocr"], timings["extract"])
        return result, timings

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _new_executor(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=self._workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=ocr_worker.load,
            initargs=(self._cpu_threads,),
        )

    async def _run(self, fn, *args):
        executor = self._executor
        try:
            return await asyncio.get_running_loop().run_in_executor(executor, fn, *args)
        except BrokenProcessPool:
            # A worker died (e.g. a native crash in Paddle); the pool is unusable after that.
            if self._executor is executor:
                logger.error("an OCR worker died; restarting the OCR pool")
                executor.shutdown(wait=False, cancel_futures=True)
                self._executor = self._new_executor()
            raise


class TransactionReplyProvider:
    """Reply to a receipt image with the transaction details read from it, and store them."""

    def __init__(self, channel: Channel, reader: ReceiptReader, store: TransactionStore):
        self._channel = channel
        self._reader = reader
        self._store = store

    async def reply(self, event: ChannelEvent) -> Optional[str]:
        if not isinstance(event, ImageMessageReceived):
            return None
        # Downloads run as soon as the webhook arrives; the reader queues images
        # until an OCR worker is free.
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
