import json
from pathlib import Path
from transformers import pipeline
import torch

# Check if GPU available for faster processing
device = 0 if torch.cuda.is_available() else -1

# Initialize NER pipeline with multilingual model
# XLM-RoBERTa supports 100+ languages including Arabic and English
ner_pipeline = pipeline(
    "token-classification",
    model="xlm-roberta-large-finetuned-conll03-english",
    device=device,
    aggregation_strategy="simple"
)

SYNTHETIC_DIR = Path(__file__).parent / "synthetic"


def extract_and_recognize_entities(text_file_path):
    """Extract text from paddle OCR file and apply NER"""

    with open(text_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        return {"text": text, "entities": []}

    # Apply NER
    entities = ner_pipeline(text)

    return {
        "text": text,
        "entities": entities
    }


def format_entities_for_display(result):
    """Format entities in a readable way"""
    print(f"\n{'='*60}")
    print(f"EXTRACTED TEXT:\n{result['text']}")
    print(f"\n{'-'*60}")
    print("NAMED ENTITIES RECOGNIZED:")
    print(f"{'-'*60}")

    if not result['entities']:
        print("No entities found")
        return

    for entity in result['entities']:
        print(f"  • {entity['word']:<30} → {entity['entity_group']:<15} (confidence: {entity['score']:.3f})")


def main():
    """Process all paddle OCR text files"""
    paddle_files = sorted(SYNTHETIC_DIR.glob("*.paddleocr.txt"))

    if not paddle_files:
        print(f"No paddle OCR text files found in {SYNTHETIC_DIR}")
        return

    print(f"Found {len(paddle_files)} paddle OCR file(s)")

    for paddle_file in paddle_files:
        print(f"\n\n{'#'*60}")
        print(f"Processing: {paddle_file.name}")
        print(f"{'#'*60}")

        result = extract_and_recognize_entities(paddle_file)
        format_entities_for_display(result)

        # Save results to JSON
        output_file = paddle_file.with_stem(paddle_file.stem + ".ner")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Results saved to: {output_file.name}")


if __name__ == "__main__":
    main()
