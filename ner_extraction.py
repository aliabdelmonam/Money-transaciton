import json
import re
import sys
from pathlib import Path

from paddle_ocr import create_ocr, extract_text

import torch
from transformers import pipeline
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path(__file__).parent / "ner_output"
MIN_SCORE = 0.5

# "first" aggregates at word level, so a word like "CIB" is never split into "C" + "IB"
ner_pipeline = pipeline(
    "token-classification",
    model="xlm-roberta-large-finetuned-conll03-english",
    device=0 if torch.cuda.is_available() else -1,
    aggregation_strategy="first",
)

MODEL_LABELS = {"PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "LOCATION", "MISC": "MISC"}

CURRENCY = r"(?:EGP|USD|EUR|SAR|AED|LE|جنيه(?:\s+مصري)?|ج\.م)"
MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"

# Order matters: earlier patterns win when spans overlap.
PATTERNS = [
    ("EMAIL", r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*"),
    ("DATE", rf"\b\d{{1,2}}\s+{MONTHS}\s+\d{{4}}(?:\s+\d{{1,2}}:\d{{2}}(?:\s*[AP]M)?)?"),
    ("DATE", r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}(?:\s+\d{1,2}:\d{2}(?:\s*[AP]M)?)?"),
    ("PHONE", r"(?<!\d)(?:\+?20|0)1[0125]\d{8}(?!\d)"),
    ("ACCOUNT_NUMBER", r"\b\d{3,4}(?:-[0-9A-Z]{1,4}){2,}\b"),
    ("REFERENCE", r"\b(?:[A-Z0-9]{2,}(?:-[A-Z0-9]+){1,}|[0-9]{10,})\b"),
    ("BANK", r"(?<!\w)(?:CIB|NBE|QNB|HSBC|AAIB|ADIB|NBK|Banque Misr|Banque du Caire|بنك مصر|البنك الأهلي(?: المصري)?|بنك القاهرة|البنك التجاري الدولي)(?!\w)"),
    ("MONEY", rf"\d[\d,]*(?:\.\d+)?\s*{CURRENCY}(?!\w)"),
    ("CURRENCY", rf"(?<!\w){CURRENCY}(?!\w)"),
]


def overlaps(start, end, spans):
    return any(start < e and s < end for s, e in spans)


def regex_entities(text):
    entities = []
    for label, pattern in PATTERNS:
        for m in re.finditer(pattern, text):
            if not overlaps(m.start(), m.end(), [(e["start"], e["end"]) for e in entities]):
                entities.append({"label": label, "start": m.start(), "end": m.end(), "score": 1.0, "source": "regex"})
    return entities


def model_entities(text, taken):
    # Run per OCR line: each line is its own field, and multi-line input misaligns the aggregated offsets.
    entities = []
    for line in re.finditer(r"[^\n]+", text):
        for ent in ner_pipeline(line.group()):
            start, end = line.start() + ent["start"], line.start() + ent["end"]
            if ent["score"] < MIN_SCORE or overlaps(start, end, taken):
                continue
            entities.append({
                "label": MODEL_LABELS.get(ent["entity_group"], ent["entity_group"]),
                "start": start,
                "end": end,
                "score": round(float(ent["score"]), 4),
                "source": "model",
            })
    return entities


def extract_entities(text):
    entities = regex_entities(text)
    entities += model_entities(text, [(e["start"], e["end"]) for e in entities])
    entities.sort(key=lambda e: e["start"])
    for e in entities:
        e["text"] = " ".join(text[e["start"]:e["end"]].split())
    return entities


def main():
    image_path = Path(__file__).parent / "synthetic/Screenshot 2026-09-24 153123.png"

    if not image_path.is_file():
        sys.exit(f"Image not found: {image_path}")

    text = extract_text(create_ocr(), image_path)
    entities = extract_entities(text)

    print(f"--- {image_path.name} ---")
    for e in entities:
        print(f"{e['text']:<30} --> {e['label']}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / f"{image_path.stem}.ner.json"
    output_file.write_text(
        json.dumps({"image": str(image_path), "text": text, "entities": entities}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
