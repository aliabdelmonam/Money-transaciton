"""Latency benchmark: every image in synthetic/ through the real receipt request path.

    python tests/perf/latency_benchmark.py

Edit the settings below the imports to change what it runs.

One "request" is what the bot does after the webhook has returned its 200, minus
the two calls to Meta (downloading the media, sending the reply): store the
pending row -> save the image -> PaddleOCR -> extractor -> store the result ->
format the reply. It uses the real ReceiptReader, TransactionReplyProvider and
TransactionStore, on a throwaway SQLite database.

Three phases:
  startup     model load time and the first (cold) request
  sequential  every image, one request at a time, RUNS times: the service time
  burst       every image sent at once, as when a user forwards a batch: the
              latency users see while requests queue for the OCR workers

Not collected by pytest (the name does not start with test_): it loads the OCR
models and takes minutes.
"""
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]  # paddle_ocr lives at the root, the app under src/

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from api.receipts import ReceiptReader, TransactionReplyProvider  # noqa: E402
from channels.events.messages import ImageMessageReceived  # noqa: E402
from channels.models.attachment import Attachment, AttachmentType  # noqa: E402
from channels.models.media import InboundMedia  # noqa: E402
from channels.models.user import User  # noqa: E402
from db import Database  # noqa: E402
from db.migrate import upgrade  # noqa: E402
from db.transactions import TransactionStore  # noqa: E402

try:
    import psutil
except ImportError:  # memory numbers are skipped without it
    psutil = None

# ---------------------------------------------------------------- settings

IMAGES_DIR = ROOT / "synthetic"   # folder of receipt images
RUNS = 1                          # sequential passes over every image; more gives run-to-run jitter
WARMUP = 1                        # unmeasured requests after the cold one
WORKERS = 4                       # OCR_WORKERS: OCR processes reading images in parallel
SLO_SECONDS = 5.0                 # latency target
BURST = True                      # also send every image at once
JSON_OUTPUT: Optional[Path] = None  # e.g. ROOT / "output" / "latency.json" to save every number

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STAGES = ("db_insert", "ocr", "extract", "db_update", "other")
NOT_FIELDS = ("evidence", "unmapped")


@dataclass
class Image:
    path: Path
    data: bytes
    width: int
    height: int

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6


@dataclass
class Request:
    id: str
    image: str
    phase: str
    megapixels: float
    submitted: float = 0.0
    ocr_started: Optional[float] = None
    finished: float = 0.0
    stages: dict = field(default_factory=dict)
    cpu: float = 0.0          # CPU seconds, OCR workers included; only meaningful one request at a time
    ocr_lines: int = 0
    fields: list = field(default_factory=list)
    read: bool = False        # the bot replied with transaction details
    error: Optional[str] = None
    db_errors: list = field(default_factory=list)  # the app logs these and carries on

    @property
    def total(self) -> float:
        return self.finished - self.submitted

    @property
    def wait(self) -> float:
        """Time from arrival until OCR started on this image (DB insert + queueing)."""
        return (self.ocr_started or self.finished) - self.submitted

    def stage(self, name: str) -> float:
        if name == "other":  # file write, thread hops, formatting, semaphore/queue waits
            return self.total - sum(self.stages.values())
        return self.stages.get(name, 0.0)

    def to_dict(self) -> dict:
        return {**asdict(self), "total": self.total, "wait": self.wait,
                "stages": {name: self.stage(name) for name in STAGES}}


class Probe:
    """Record the OCR and extraction time of each request, as measured inside the OCR worker."""

    def __init__(self):
        self.requests: dict[str, Request] = {}

    def install(self, reader: ReceiptReader) -> None:
        async def timed_read(data, filename):
            result, timings = await reader.read_timed(data, filename)
            request = self.requests[Path(filename).stem]  # images are named <request id>.<ext>
            request.stages["ocr"], request.stages["extract"] = timings["ocr"], timings["extract"]
            # The worker's clock is not ours; place the OCR start by working back from now.
            request.ocr_started = time.perf_counter() - timings["ocr"] - timings["extract"]
            request.ocr_lines = timings["lines"]
            request.fields = sorted(k for k, v in result.items() if v and k not in NOT_FIELDS)
            return result

        reader.read = timed_read


class TimedStore:
    """TransactionStore that records how long each database write took."""

    def __init__(self, store: TransactionStore, probe: Probe):
        self._store = store
        self._probe = probe

    async def record_received(self, event):
        return await self._timed("db_insert", event, self._store.record_received(event))

    async def record_result(self, event, media, result):
        return await self._timed("db_update", event, self._store.record_result(event, media, result))

    async def record_failure(self, event, error, media=None):
        return await self._timed("db_update", event, self._store.record_failure(event, error, media))

    async def _timed(self, stage, event, write):
        request = self._probe.requests[event.provider_message_id]
        start = time.perf_counter()
        try:
            return await write
        except Exception as error:
            request.db_errors.append(f"{stage}: {error!r}")
            raise
        finally:
            request.stages[stage] = time.perf_counter() - start


class FakeChannel:
    """Hands back the image bytes carried on the attachment instead of downloading them from Meta."""

    name = "whatsapp"

    async def fetch_media(self, attachment: Attachment) -> InboundMedia:
        return InboundMedia(attachment.data, attachment.mime_type, attachment.filename)


def family(process) -> list:
    """This process and its OCR workers (which may exit between listing and reading them)."""
    alive = []
    for member in [process, *process.children(recursive=True)]:
        try:
            member.memory_info()
            alive.append(member)
        except psutil.Error:
            pass
    return alive


def cpu_seconds() -> float:
    """CPU time used so far by this process and its OCR workers."""
    if psutil is None:
        return time.process_time()  # misses the workers
    total = 0.0
    for member in family(psutil.Process()):
        try:
            times = member.cpu_times()
            total += times.user + times.system
        except psutil.Error:
            pass
    return total


class MemorySampler:
    """Track the peak resident memory of this process and its OCR workers while the benchmark runs."""

    def __init__(self, interval: float = 0.05):
        self._process = psutil.Process() if psutil else None
        self._interval = interval
        self._task = None
        self.peak = 0

    def now(self) -> Optional[int]:
        if self._process is None:
            return None
        rss = sum(p.memory_info().rss for p in family(self._process))
        self.peak = max(self.peak, rss)
        return rss

    async def _run(self):
        while True:
            self.now()
            await asyncio.sleep(self._interval)

    def start(self):
        if self._process is not None:
            self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)


# ---------------------------------------------------------------- running requests

def load_images(folder: Path) -> list[Image]:
    images = []
    for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        data = path.read_bytes()
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        height, width = decoded.shape[:2] if decoded is not None else (0, 0)
        images.append(Image(path, data, width, height))
    return images


async def send(provider, store, probe, image: Image, request_id: str, phase: str) -> Request:
    request = Request(request_id, image.path.name, phase, image.megapixels)
    probe.requests[request_id] = request
    event = ImageMessageReceived(
        channel="whatsapp",
        conversation_id="benchmark",
        provider_message_id=request_id,
        sender=User(id="benchmark"),
        attachment=Attachment(
            type=AttachmentType.IMAGE,
            data=image.data,
            filename=request_id + image.path.suffix.lower(),
            mime_type="image/png" if image.path.suffix.lower() == ".png" else "image/jpeg",
        ),
    )
    cpu = cpu_seconds()
    request.submitted = time.perf_counter()
    # Same order as RecordingResponder.handle -> TransactionReplyProvider.reply.
    try:
        await store.record_received(event)
    except Exception:
        pass  # RecordingResponder logs a failed insert and answers anyway; TimedStore counted it
    try:
        reply = await provider.reply(event)
    except Exception as error:
        request.error = repr(error)
        reply = None
    request.finished = time.perf_counter()
    request.cpu = cpu_seconds() - cpu
    request.read = reply is not None
    return request


async def benchmark() -> dict:
    images = load_images(IMAGES_DIR)
    if not images:
        sys.exit(f"No images in {IMAGES_DIR}")
    memory = MemorySampler()

    with tempfile.TemporaryDirectory(prefix="latency-bench-") as tmp:
        url = f"sqlite+aiosqlite:///{(Path(tmp) / 'bench.db').as_posix()}"
        await asyncio.to_thread(upgrade, url)
        database = Database(url)
        reader = ReceiptReader(Path(tmp) / "inbound", WORKERS)
        probe = Probe()
        memory.start()
        try:
            rss_before = memory.now()
            print("loading OCR models...", flush=True)
            start = time.perf_counter()
            await reader.start()
            model_load = time.perf_counter() - start
            rss_loaded = memory.now()
            probe.install(reader)

            store = TimedStore(TransactionStore(database), probe)
            provider = TransactionReplyProvider(FakeChannel(), reader, store)

            cold = await send(provider, store, probe, images[0], "cold-000", "cold")
            for w in range(WARMUP):  # not measured
                await send(provider, store, probe, images[w % len(images)], f"warmup-{w:03d}", "warmup")

            sequential = []
            wall = time.perf_counter()
            for run in range(RUNS):
                for i, image in enumerate(images):
                    print(f"\rsequential  run {run + 1}/{RUNS}  image {i + 1}/{len(images)}  ",
                          end="", flush=True)
                    sequential.append(await send(provider, store, probe, image, f"seq-{run}-{i:03d}", "sequential"))
            sequential_wall = time.perf_counter() - wall
            print()

            burst, burst_wall = [], 0.0
            if BURST:
                print(f"burst       {len(images)} images at once...", flush=True)
                wall = time.perf_counter()
                burst = await asyncio.gather(*(
                    send(provider, store, probe, image, f"burst-{i:03d}", "burst") for i, image in enumerate(images)
                ))
                burst_wall = time.perf_counter() - wall
        finally:
            await memory.stop()
            reader.close()
            await database.dispose()

    return {
        "config": {
            "images": len(images), "runs": RUNS, "warmup": WARMUP, "workers": WORKERS,
            "slo_seconds": SLO_SECONDS, "cpu_count": os.cpu_count(), "python": sys.version.split()[0],
        },
        "startup": {"model_load": model_load, "cold_request": cold.to_dict(),
                    "rss_before_load": rss_before, "rss_after_load": rss_loaded, "rss_peak": memory.peak or None},
        "images": images,
        "sequential": sequential,
        "sequential_wall": sequential_wall,
        "burst": burst,
        "burst_wall": burst_wall,
    }


# ---------------------------------------------------------------- statistics and report

PERCENTILES = (50, 90, 95, 99)


def describe(values) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {}
    out = {"count": int(values.size), "min": values.min(), "mean": values.mean(),
           "stdev": values.std(ddof=1) if values.size > 1 else 0.0, "max": values.max()}
    out.update({f"p{q}": np.percentile(values, q) for q in PERCENTILES})
    return {k: float(v) for k, v in out.items()}


def ms(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    value = seconds * 1000
    return f"{value:,.0f}" if value >= 100 else f"{value:.1f}"


def mb(size: Optional[int]) -> str:
    return "n/a" if size is None else f"{size / 2**20:,.0f} MB"


def stats_table(rows: list[tuple[str, dict]], share_of: Optional[float] = None) -> None:
    columns = ("min", "mean", "p50", "p90", "p95", "p99", "max", "stdev")
    header = f"  {'(ms)':<12}" + "".join(f"{c:>9}" for c in columns) + ("    share" if share_of else "")
    print(header)
    for name, s in rows:
        line = f"  {name:<12}" + "".join(f"{ms(s.get(c)):>9}" for c in columns)
        if share_of:
            line += f"  {s['mean'] / share_of:>6.1%}"
        print(line)


def summarize(requests: list[Request], wall: float, slo: float) -> dict:
    ok = [r for r in requests if r.error is None]
    totals = [r.total for r in requests]
    summary = {
        "requests": len(requests),
        "errors": len(requests) - len(ok),
        "db_errors": sum(len(r.db_errors) for r in requests),
        "wall_seconds": wall,
        "throughput_rps": len(requests) / wall if wall else None,
        "total": describe(totals),
        "wait_before_ocr": describe([r.wait for r in requests]),
        "stages": {name: describe([r.stage(name) for r in requests]) for name in STAGES},
        "within_slo": sum(t <= slo for t in totals) / len(totals),
        "read_rate": sum(r.read for r in requests) / len(requests),
        "cpu_seconds": describe([r.cpu for r in requests]),
        "ms_per_megapixel": describe([r.stage("ocr") * 1000 / r.megapixels for r in requests if r.megapixels]),
    }
    return summary


def per_image(requests: list[Request], images: list[Image]) -> list[dict]:
    rows = []
    for image in images:
        runs = [r for r in requests if r.image == image.path.name]
        totals = [r.total for r in runs]
        rows.append({
            "image": image.path.name,
            "kb": len(image.data) / 1024,
            "resolution": f"{image.width}x{image.height}",
            "megapixels": image.megapixels,
            "ocr_lines": runs[-1].ocr_lines,
            "fields": runs[-1].fields,
            "read": all(r.read for r in runs),
            "errors": sum(r.error is not None for r in runs),
            "total_mean": float(np.mean(totals)),
            "total_stdev": float(np.std(totals, ddof=1)) if len(totals) > 1 else 0.0,
            "ocr_mean": float(np.mean([r.stage("ocr") for r in runs])),
            "extract_mean": float(np.mean([r.stage("extract") for r in runs])),
        })
    return sorted(rows, key=lambda row: row["total_mean"], reverse=True)


def report(result: dict) -> dict:
    config, startup, images = result["config"], result["startup"], result["images"]
    slo = config["slo_seconds"]
    cold = startup["cold_request"]
    rule = "=" * 96

    print(f"\n{rule}\nLATENCY BENCHMARK  {config['images']} images x {config['runs']} runs"
          f"  |  {config['cpu_count']} CPUs  |  Python {config['python']}\n{rule}")

    print("\nSTARTUP")
    print(f"  model load               {startup['model_load']:8.2f} s")
    print(f"  first request (cold)     {cold['total']:8.2f} s   ({cold['image']}, OCR {ms(cold['stages']['ocr'])} ms)")
    print(f"  memory                   before load {mb(startup['rss_before_load'])}"
          f" | after load {mb(startup['rss_after_load'])} | peak {mb(startup['rss_peak'])}"
          + ("" if psutil else "   (pip install psutil)"))

    seq = summarize(result["sequential"], result["sequential_wall"], slo)
    total = seq["total"]
    print(f"\nSEQUENTIAL  one request at a time, {seq['requests']} requests"
          f" (service time, no queueing)")
    stats_table([("total", total)] + [(name, seq["stages"][name]) for name in STAGES], share_of=total["mean"])
    cpu = seq["cpu_seconds"]["mean"]
    print(f"\n  avg latency              {total['mean']:8.2f} s    p95 {total['p95']:.2f} s    p99 {total['p99']:.2f} s")
    print(f"  throughput               {seq['throughput_rps']:8.2f} req/s   ({3600 * seq['throughput_rps']:,.0f} receipts/hour)")
    print(f"  within SLO ({slo:g} s)       {seq['within_slo']:8.1%}")
    print(f"  tail ratio p95 / p50     {total['p95'] / total['p50']:8.2f} x")
    warm = [r.total for r in result["sequential"] if r.image == cold["image"]]
    print(f"  cold vs warm             {cold['total'] / np.mean(warm):8.2f} x   (same image, first request vs later ones)")
    print(f"  CPU per request          {cpu:8.2f} s    ({cpu / total['mean']:.1f} of {config['cpu_count']} cores busy on average)")
    print(f"  OCR per megapixel        {seq['ms_per_megapixel']['mean']:8.0f} ms/MP")
    print(f"  read rate                {seq['read_rate']:8.1%}   (requests answered with transaction details)")
    print(f"  errors                   {seq['errors']:8d}    db write errors {seq['db_errors']}")

    images_rows = per_image(result["sequential"], images)
    print(f"\nPER IMAGE  mean of {config['runs']} runs, slowest first")
    print(f"  {'image':<44}{'KB':>6}{'resolution':>12}{'lines':>6}{'fields':>7}"
          f"{'total ms':>10}{'+/-':>6}{'ocr ms':>8}{'extr ms':>8}  read")
    for row in images_rows:
        print(f"  {row['image'][:43]:<44}{row['kb']:>6.0f}{row['resolution']:>12}{row['ocr_lines']:>6}"
              f"{len(row['fields']):>7}{ms(row['total_mean']):>10}{ms(row['total_stdev']) if config['runs'] > 1 else '-':>6}"
              f"{ms(row['ocr_mean']):>8}{ms(row['extract_mean']):>8}  {'yes' if row['read'] else 'NO'}")
    unread = [row["image"] for row in images_rows if not row["read"]]
    if unread:
        print(f"  not read: {', '.join(unread)}")

    burst = None
    if result["burst"]:
        burst = summarize(result["burst"], result["burst_wall"], slo)
        print(f"\nBURST  all {burst['requests']} images at once, workers={config['workers']}"
              f" (what users see when requests queue)")
        stats_table([("total", burst["total"]), ("wait", burst["wait_before_ocr"]), ("ocr", burst["stages"]["ocr"])])
        print(f"\n  last reply after         {burst['wall_seconds']:8.2f} s")
        print(f"  throughput               {burst['throughput_rps']:8.2f} req/s")
        print(f"  within SLO ({slo:g} s)       {burst['within_slo']:8.1%}")
        print(f"  queueing share of p95    {burst['wait_before_ocr']['p95'] / burst['total']['p95']:8.1%}")
        print(f"  errors                   {burst['errors']:8d}    db write errors {burst['db_errors']}")
    print(rule)

    return {
        "config": config,
        "startup": startup,
        "sequential": seq,
        "per_image": images_rows,
        "burst": burst,
        "requests": [r.to_dict() for r in result["sequential"] + list(result["burst"])],
    }


def main() -> None:
    # The app logs failed DB writes with a traceback; they are counted in the report instead.
    logging.getLogger("api").setLevel(logging.CRITICAL)

    summary = report(asyncio.run(benchmark()))
    if JSON_OUTPUT:
        JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        JSON_OUTPUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved {JSON_OUTPUT}")


if __name__ == "__main__":
    main()
