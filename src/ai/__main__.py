"""Batch run over a folder of receipts (replacement for script_ocr.py main).

    python -m src.ai --input ./synthetic --output ./output --engine paddle
"""

import argparse
import csv
import glob
import json
import logging
import os

from .adapters.ocr import ENGINES
from src.config.ai import AIConfig
from .extraction.models import Party
from .imaging.debug import draw_debug
from .pipeline import ReceiptPipeline

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
CSV_COLUMNS = ["file", "provider", "status", "amount", "fees", "total", "currency", "reference",
               "date", "date_iso", "note",
               *(f"{side}_{k}" for side in ("sender", "receiver") for k in Party.keys()),
               "other_phones", "derived", "warnings"]


def flatten(res: dict) -> dict:
    row = {k: res.get(k) for k in CSV_COLUMNS if k in res}
    for side in ("sender", "receiver"):
        for k in Party.keys():
            row[f"{side}_{k}"] = res[side].get(k)
    for k in ("other_phones", "derived", "warnings"):
        row[k] = " | ".join(res.get(k) or [])
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Egyptian InstaPay / e-wallet receipt reader")
    ap.add_argument("--input", default="./synthetic")
    ap.add_argument("--output", default="./output")
    ap.add_argument("--engine", choices=ENGINES, default="paddle")
    ap.add_argument("--cpu", action="store_true", help="disable GPU")
    ap.add_argument("--no-debug", action="store_true", help="skip debug images")
    ap.add_argument("--no-text", action="store_true", help="skip raw text files")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("ai")

    files = sorted(p for p in glob.glob(os.path.join(args.input, "*")) if p.lower().endswith(IMAGE_EXTENSIONS))
    if not files:
        log.error("No images found in %s", os.path.abspath(args.input))
        return
    for sub in ("json", "text", "debug"):
        os.makedirs(os.path.join(args.output, sub), exist_ok=True)

    pipeline = ReceiptPipeline(config=AIConfig(engine=args.engine, use_gpu=not args.cpu))
    results = []
    for i, path in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            analysis = pipeline.analyze_file(path)
        except Exception as e:
            log.error("[%d/%d] %s: %s", i, len(files), name, e)
            continue
        res = {"file": os.path.basename(path), **analysis.to_dict()}
        results.append(res)
        with open(os.path.join(args.output, "json", name + ".json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        if not args.no_text:
            with open(os.path.join(args.output, "text", name + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(res["text_lines"]))
        if not args.no_debug:
            draw_debug(analysis.image, analysis.tokens, os.path.join(args.output, "debug", name + ".png"))
        r = analysis.receipt
        log.info("[%d/%d] %s: %s | amount=%s ref=%s | from=%s -> to=%s", i, len(files), name, r.provider,
                 r.amount, r.reference, r.sender.name or r.sender.phone, r.receiver.name or r.receiver.phone)

    summary = [{k: v for k, v in r.items() if k != "tokens"} for r in results]
    with open(os.path.join(args.output, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output, "results.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(flatten(r))
    log.info("Done. Results in %s", os.path.abspath(args.output))


if __name__ == "__main__":
    main()
