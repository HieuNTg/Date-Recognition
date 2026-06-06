# DateReg — Expiry Date Recognition System

An end-to-end deep learning pipeline that detects and reads expiry dates from product images, built with **YOLOv8** for detection and a **CTC-based OCR** model for text recognition.

<p align="center">
  <img width="1024" alt="Demo" src="https://github.com/HieuNTg/Date-Recognition/assets/96096473/23da4d31-45d7-4824-96bd-3b2aa25b090f">
</p>

## Features

- **Object Detection** — YOLOv8 locates date regions on product packaging
- **OCR** — Custom CTC model recognizes date text from cropped regions
- **Smart Parsing** — Handles various date formats, strips prefixes (EXP, BB, MFG, NSX, HSD), fixes common OCR misreads
- **Expiry Evaluation** — Color-coded status: green (valid), orange (expiring soon), red (expired)
- **Visual Feedback** — Bounding boxes with confidence scores drawn directly on the image
- **Configurable** — All model paths, thresholds, and parameters managed via `configs/config.yaml`

## Architecture

```
                    ┌─────────────┐
                    │ Input Image │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │   Pre-      │
                    │ Processing  │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │   Detect    │
                    │   YOLOv8    │
                    └──────┬──────┘
                           ▼
            ┌──────────── CRNN ────────────┐
            │                              │
            │  ┌────────┐    ┌───────────┐    ┌──────────┐
            │  │  CNN   │──▶ │   RNN    │ ──▶│ CTC Loss │
            │  │Feature │    │ (BiLSTM)  │    │          │
            │  │Extract.│    │           │    │          │
            │  └────────┘    └───────────┘    └──────────┘
            │                              │
            └──────────────┬───────────────┘
                           ▼
                    ┌─────────────┐
                    │    Text     │
                    └─────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Detection | YOLOv8 (Ultralytics) |
| OCR | TensorFlow / Keras + CTC Decoder |
| Web UI | Streamlit |
| REST API | FastAPI + Uvicorn |
| Packaging | Docker |
| Date Parsing | python-dateutil |
| Configuration | YAML |

## Project Structure

```
DateReg/
├── app.py                          # Streamlit web app
├── cli.py                          # Batch inference CLI (folder → JSON/CSV)
├── api/
│   └── main.py                     # FastAPI REST service (/predict, /health)
├── configs/
│   └── config.yaml                 # Model paths, thresholds, parameters
├── src/
│   ├── pipeline.py                 # DatePipeline — UI-agnostic orchestration
│   ├── detection/
│   │   └── detector.py             # YOLODetector class
│   ├── recognition/
│   │   └── ocr.py                  # OCRRecognizer class (CTC)
│   └── utils/
│       └── date_parser.py          # DateParser class
├── models/
│   ├── yolo/best.pt                # Trained YOLOv8 weights
│   └── ocr/best_model_new.h5       # Trained CTC-OCR weights
├── notebooks/
│   ├── train_yolo.ipynb            # YOLOv8 training notebook
│   └── train_ocr.ipynb             # OCR training notebook
├── tests/
│   └── test_date_parser.py         # Unit tests (no model/data needed)
├── Dockerfile                      # Container image (serves the API)
├── .dockerignore
├── requirements.txt                # Runtime deps (incl. API)
├── requirements-dev.txt            # + pytest / httpx
└── packages.txt                    # System dependencies (libgl1)
```

All three entry points — web app, REST API, and CLI — drive inference through
the single `DatePipeline` class in `src/pipeline.py`, so behaviour is identical
everywhere and the model-loading/orchestration logic lives in exactly one place.

## Dataset

| Dataset | Total | Train | Val | Test |
|---------|-------|-------|-----|------|
| Date-Synth (text images) | 128,510 | 89,957 | 25,702 | 12,851 |
| Products-Synth (product images) | 11,860 | 8,300 | 2,371 | 1,187 |

## Results

**Detection (YOLOv8):**

|  | Precision | Recall | mAP50 | mAP50-95 |
|---|:-:|:-:|:-:|:-:|
| **Training** | 0.969 | 0.963 | 0.981 | 0.862 |
| **Test** | 0.960 | 0.963 | 0.976 | 0.874 |

**Text Recognition (CTC-OCR):**

| Metric | Score |
|---|:-:|
| **CER** (Character Error Rate) | 0.05 |
| **WER** (Word Error Rate) | 0.19 |

## Engineering Decisions & Trade-offs

The interesting part of this project wasn't training a model — it was the
design choices that make a noisy, real-world OCR problem tractable.

- **Two-stage detect-then-read, not end-to-end.** Expiry dates occupy a tiny
  fraction of a product photo and sit on cluttered packaging. Running OCR over
  the full image floods the recognizer with distractor text (ingredients,
  branding, barcodes). A YOLOv8 stage first isolates the date region, so the
  CTC model only ever sees a tight, relevant crop — far higher accuracy than a
  single end-to-end network, and each stage can be debugged and retrained
  independently.

- **CTC over a fixed-vocabulary classifier.** Dates are variable-length
  sequences (`01/26`, `15 JUN 2026`, `MFG2024`). CTC lets the model emit a
  variable-length string from a single forward pass with no per-character
  segmentation — the natural fit for sequence recognition, and why the OCR head
  is a CRNN (CNN features → BiLSTM → CTC) rather than a classifier.

- **Constrained character set (44 classes).** The vocabulary is deliberately
  limited to the glyphs that actually appear in dates (digits, month letters,
  `/`). A smaller output space means a smaller, faster model and fewer
  confusable classes — at the documented cost that `.` and `-` separators
  aren't recognised (see *Known Limitations*). A conscious accuracy-vs-coverage
  trade-off, not an oversight.

- **Rotated input (`transpose` before resize).** The CTC model reads along the
  width axis, so each crop is transposed to `224×64` before inference to align
  the text's reading direction with the time axis the BiLSTM unrolls over.

- **Heuristic post-processing instead of a bigger model.** Two cheap rules
  recover most field errors without retraining: a prefix stripper
  (`EXP`/`BB`/`MFG`/`NSX`/`HSD`) and an `O→0` repair for the single most common
  OCR confusion. When several dates are detected, the **latest** is chosen as
  the expiry — the domain-correct disambiguation when both MFG and EXP appear.

- **Stateless, in-memory pipeline.** The detector returns crops as PIL objects
  rather than writing fixed-name files to a shared `temp/` dir. This removed a
  race condition (concurrent requests clobbering each other's crops) and let a
  single loaded `DatePipeline` instance safely serve the web app, the REST API,
  and batch CLI jobs concurrently.

- **Config-driven, not hard-coded.** Every weight path, threshold, input size
  and warning window lives in `configs/config.yaml`, so tuning the deployment
  never touches Python.

## Quick Start

```bash
git clone https://github.com/HieuNTg/Date-Recognition.git
cd Date-Recognition
pip install -r requirements.txt
```

Trained weights ship with the repo (`models/`), so it runs out of the box.

## Deployment & Usage

The same pipeline is exposed three ways — pick the one that fits.

**1. Web app (Streamlit)** — interactive demo with bounding-box overlay:

```bash
streamlit run app.py
```

**2. REST API (FastAPI)** — for service-to-service integration:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Swagger docs at http://localhost:8000/docs

curl -F "file=@product.jpg" http://localhost:8000/predict
```

```jsonc
{
  "date": "2026-06-01",
  "status": "valid",
  "days_remaining": 360,
  "detections": [
    { "text": "EXP 01/06/2026", "confidence": 0.97, "bbox": [120, 84, 318, 142] }
  ]
}
```

**3. Batch CLI** — score a whole folder of images to JSON or CSV:

```bash
python cli.py path/to/images/ --out results.json
python cli.py path/to/images/ --out results.csv --format csv
```

**4. Docker** — containerized API, weights baked in:

```bash
docker build -t datereg .
docker run -p 8000:8000 datereg
```

## Testing

The date-parsing logic is fully unit-tested and needs no model or dataset:

```bash
pip install -r requirements-dev.txt
pytest
```

## Configuration

All parameters are centralized in `configs/config.yaml`:

```yaml
model:
  yolo:
    confidence: 0.25    # Detection confidence threshold
    padding: 5          # Bounding box padding (px)
  ocr:
    img_width: 224      # OCR input width
    img_height: 64      # OCR input height

date_parser:
  warning_days: 30      # Days before expiry to show warning
```

## Known Limitations

- OCR character set does not include `.` and `-` separators (would require retraining)
- Date format parsing defaults to `dateutil` heuristics — may misinterpret ambiguous formats (e.g., `01/02/2026`)
