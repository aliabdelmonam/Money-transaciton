# Money Transaction OCR Comparison

A Python project that compares the performance of three different OCR (Optical Character Recognition) engines for extracting text from money transaction screenshots. This project is designed to evaluate and benchmark OCR accuracy on financial transaction images, supporting both English and Arabic text extraction.

## Overview

This project implements three popular OCR engines to process transaction images and extract text:

- **PaddleOCR** - Baidu's open-source OCR framework (PP-OCRv6 detection, Arabic + English PP-OCRv5 recognition)
- **EasyOCR** - A simple yet powerful OCR library
- **Tesseract OCR** - Google's open-source OCR engine

Each engine processes the same set of synthetic transaction images and outputs extracted text to separate files for easy comparison and analysis.

## Features

- 🎯 **Multi-OCR Comparison** - Compare results from three different OCR engines
- 🌍 **Multilingual Support** - Supports Arabic and English text extraction
- 📸 **Batch Processing** - Process multiple images in a single run
- 📊 **Confidence Filtering** - Configurable confidence thresholds to filter low-quality extractions
- 🔤 **UTF-8 Support** - Full Unicode support for Arabic and other languages

## Project Structure

```
money-transaction/
├── README.md                      # This file
├── paddle_ocr.py                  # PaddleOCR implementation (same interface as tesseract_ocr.py)
├── paddle_engine/                 # its two-recognizer line reading and masked-name rebuilding
├── easy_ocr.py                    # EasyOCR implementation
├── tesseract_ocr.py               # Tesseract OCR implementation (same interface as paddle_ocr.py)
├── tesseract_engine/              # its image preparation, line finding and line readers
├── transaction_extractor.py       # Structured field extraction from receipts (uses PaddleOCR)
├── transaction_config.py          # Label / provider / currency vocabulary for the extractor
├── transaction_output/            # One JSON result per image
└── synthetic/                     # Test images directory
    ├── InstaPay_success.jpg
    ├── InstaPay_success_ar.jpg
    └── [extracted text files]     # Output files from OCR processing
```

## Requirements

### System Requirements

- **Python 3.7+**
- **Tesseract OCR** (for tesseract_ocr.py)
  - Windows: Download installer from [Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)
  - Default installation path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

### Python Dependencies

Install required packages using pip:

```bash
pip install paddleocr easyocr pytesseract pillow
```

For optimal performance, you can also install optional dependencies:

```bash
pip install numpy opencv-python
```

## Installation & Setup

### 1. Clone or Download the Project

```bash
cd money-transaction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install paddleocr easyocr pytesseract pillow
```

### 3. Install Tesseract (if using tesseract_ocr.py)

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Run the installer (default path: `C:\Program Files\Tesseract-OCR`)

**macOS:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

## Usage

### Extracting transaction data

```bash
python transaction_extractor.py                      # every image in synthetic/
python transaction_extractor.py receipt.png folder/  # specific images or folders
```

Produces provider, status, amount / fees / total, currency, sender and receiver
(name, phone, account, bank), reference, date-time and note for each receipt, saved to
`transaction_output/<image>.json` together with the evidence (which label each value came from)
and any lines it could not map.

It works from the layout instead of per-app templates: labels are recognised from the
Arabic/English vocabulary in `transaction_config.py` and paired with the value beside, below or
above them. To support a new app or layout, add its label wording, provider name or bank there.

### WhatsApp bot (webhook API)

Users send a receipt screenshot on WhatsApp and the bot replies with the extracted details.

1. `pip install -r requirements.txt`, then copy `.env.example` to `.env` and fill in the values.
2. Start the server from the project root:

   ```bash
   python -m uvicorn api.main:app --app-dir src --host 0.0.0.0 --port 8000
   ```

3. Expose it publicly (e.g. `ngrok http 8000`) and in the Meta dashboard set the callback URL to
   `https://<public-host>/webhooks/whatsapp`, the verify token to `CHANNELS_WHATSAPP__VERIFY_TOKEN`,
   and subscribe to the `messages` field.

| Endpoint | Purpose |
| --- | --- |
| `GET /webhooks/whatsapp` | Meta's verification handshake (`hub.challenge`) |
| `POST /webhooks/whatsapp` | Incoming updates; signature checked with `CHANNELS_WHATSAPP__WEBHOOK_SECRET` |
| `GET /health` | Liveness |
| `GET /health/whatsapp` | Graph API token check |

Every endpoint except the webhook needs the `X-API-Key` header set to `API_KEY`; without it (or with
`API_KEY` unset) the request gets a 401. The webhook can't carry that header because Meta calls it, so
the server will not start without `CHANNELS_WHATSAPP__WEBHOOK_SECRET`, and webhook calls that are not
signed with it are rejected.

### Database

Async SQLAlchemy models live in `src/db/models/`, Alembic migrations in `src/db/migrations/`.
The URL comes from `DATABASE_URL` (env or `.env`); the default is SQLite at `./data/smaia.db`.

| Table | Holds |
| --- | --- |
| `users` | One row per person per channel (`channel` + `external_id`, e.g. the WhatsApp number) |
| `messages` | Inbound and outbound messages; `(channel, provider_message_id)` is unique so retried webhooks are detected |
| `attachments` | Media on a message: the image bytes (`data`), WhatsApp media id, mime type, `sha256` |
| `transactions` | The transfer read from a receipt: provider, amount/fees/total, currency, reference, date, note, `other_data`, and the full extractor output in `raw_result` |
| `transaction_parties` | Sender and receiver of a transaction: name, phone, email, account / wallet, bank |

```bash
alembic upgrade head                                   # create / update the schema
alembic revision --autogenerate -m "add x to y"        # after changing a model, then review the file
alembic downgrade -1                                   # undo the last migration
alembic -x url=postgresql+asyncpg://... upgrade head   # target another database
python -m pytest tests/db                              # includes a check that models and migrations match
```

### Comparing OCR engines

Each OCR engine has its own script. Run them individually or all together to compare results.

### Using PaddleOCR

```bash
python paddle_ocr.py
```

**Output files:** `*.paddleocr.txt`

Features (pipeline in `paddle_engine/`):
- **Image preparation**: the Tesseract engine's `image.prepare` (dark mode inverted, photos of a
  screen cut out, straightened and evenly lit, margins cropped, text scaled to ~28 px, tilt removed)
- **Line finding**: `PP-OCRv6_medium_det` with the PaddleOCR pipeline's settings (the module's own
  defaults shrink the image to 960 px and lose small text); overlapping boxes of one line
  (`750` + `0 EGP`) merged. No textline orientation model: it flipped whole lines upside down
- **Two recognizers per line** (`recognize.py`): `arabic_PP-OCRv5_mobile_rec` (the only one that
  reads Arabic) and `en_PP-OCRv5_mobile_rec` (far better on Latin, e-mails, digits); Latin lines take
  the English reading, Arabic lines the Arabic one, mixed lines (`Living Expenses - عيدية`) each
  script from its own model
- **Masked names** (`masked.py`): asterisks found and counted by shape on the Tesseract engine's line
  blocks, erased, and the prefix and each initial read on their own; initials are read with a
  letters-only CTC whitelist (the Arabic model otherwise reads `م` as `p`) and voted across paddings
- GPU when available (~0.35 s per receipt); on the CPU detection alone takes ~10 s

### Using EasyOCR

```bash
python easy_ocr.py
```

**Output files:** `*.easyocr.txt`

Features:
- Supports both Arabic and English
- GPU acceleration available
- Good balance between speed and accuracy

### Using Tesseract OCR

```bash
python tesseract_ocr.py
```

**Output files:** `*.tesseract.txt`

Features (pipeline in `tesseract_engine/`, adapted from the PaymentOcr project):
- **Image preparation** (`image.py`): dark-mode screenshots inverted, photos of a screen cut out of
  their surround, straightened and evenly lit, grey pages whitened, wide margins cropped, the image
  scaled so text is ~28 px tall, tilt up to 10° removed
- **Line finding** (`segment.py`): OpenCV morphology instead of Tesseract's page layout, icons dropped
  by colour, card borders removed, ink measured against the local page brightness
- **Line reading** (`reader.py`): every line read alone with the English and the Arabic
  `tessdata_best` model, the more plausible script kept; phone / reference numbers, addresses and
  dates voted across scales with a character whitelist; masked names (`Gameel A O***`) rebuilt around
  their asterisks; the headline amount split by colour; mixed Arabic + English notes read with the
  combined model; small words beside big ones re-read at their own size
- **Fast**: Tesseract runs in-process through `libtesseract-5.dll` (models loaded once per thread),
  ~0.5 s per receipt; set `OCR_BACKEND=cli` to use `tesseract.exe` instead (about 5x slower)
- Line scores are 0-1 like Paddle's, but not on its scale (`MIN_CONFIDENCE = 0.2`)

## Configuration

Each script can be customized by modifying these parameters:

```python
MIN_CONFIDENCE = 0.6        # Minimum confidence threshold (0.0-1.0)
IMAGE_EXTS = {...}          # Supported image extensions
SYNTHETIC_DIR = ...         # Directory containing test images
```

### Confidence Thresholds

- **PaddleOCR & EasyOCR:** 0.0-1.0 scale (current: 0.6)
- **Tesseract:** 0-100 scale (current: 60)

Adjust these values to filter out low-confidence extractions. Higher values = stricter filtering.

## Output Format

Each OCR engine generates text output files in the same directory as the source image:

- `image.paddleocr.txt` - PaddleOCR results
- `image.easyocr.txt` - EasyOCR results
- `image.tesseract.txt` - Tesseract results
- `image.txt` - Reference/ground truth (if available)

Example output structure:
```
--- InstaPay_success.jpg ---
Your transaction has been completed successfully
TransactionID: 123456789
Amount: SAR 500.00
Date: 2024-09-24
```

## Performance Comparison

### Benchmarks

The project helps compare:

- **Accuracy** - Text extraction correctness
- **Speed** - Processing time per image
- **Language Support** - Handling of Arabic and English
- **Confidence Scores** - Reliability indication

### Running Comparisons

To compare all three engines on the same images:

```bash
python paddle_ocr.py
python easy_ocr.py
python tesseract_ocr.py
```

Then compare the `.txt` output files to evaluate accuracy and differences.

## Supported Image Formats

- JPG/JPEG
- PNG
- BMP
- WebP

## Troubleshooting

### Common Issues

**Issue:** PaddleOCR or other libraries fail to download models
```python
# This is handled automatically by setting:
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
```

**Issue:** Tesseract not found
- Verify installation path in `tesseract_ocr.py` line 10
- Update the path if installed in a different location

**Issue:** Arabic text not extracting correctly
- Ensure UTF-8 encoding: `sys.stdout.reconfigure(encoding="utf-8")`
- Check confidence threshold isn't too high
- Verify image quality is sufficient

**Issue:** Memory issues with large images
- EasyOCR: Set `gpu=False` (already configured)
- PaddleOCR: Set `enable_mkldnn=False` (already configured)

## Notes & Observations

- **Text Grouping:** Tesseract preserves document structure; PaddleOCR and EasyOCR output text sequentially
- **Arabic Support:** All three engines support Arabic, but with varying accuracy
- **Processing Speed:** PaddleOCR is typically fastest; Tesseract is slowest but most precise
- **GPU Acceleration:** EasyOCR supports GPU if available (set `gpu=True`)

## Contributing

To add support for additional languages or OCR engines:

1. Create a new script following the existing pattern
2. Implement `extract_text()` and `main()` functions
3. Ensure output files follow the naming convention: `image.{engine}.txt`
4. Update this README with configuration details

## License

This project is provided as-is for comparison and evaluation purposes.

## References

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
