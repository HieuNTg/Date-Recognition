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
Input Image → YOLOv8 (Detection) → Crop Regions → CTC-OCR (Recognition) → Date Parsing → Expiry Status
```

<p align="center">
  <img width="604" alt="Pipeline" src="https://github.com/HieuNTg/Date-Recognition/assets/96096473/b5c758ed-fcb5-4e10-a2c8-a71720b8865a">
</p>

## Tech Stack

| Component | Technology |
|-----------|------------|
| Detection | YOLOv8 (Ultralytics) |
| OCR | TensorFlow / Keras + CTC Decoder |
| Web UI | Streamlit |
| Date Parsing | python-dateutil |
| Configuration | YAML |

## Project Structure

```
DateReg/
├── app.py                          # Streamlit entry point
├── configs/
│   └── config.yaml                 # Model paths, thresholds, parameters
├── src/
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
├── .gitignore
├── requirements.txt
└── packages.txt                    # System dependencies (libgl1)
```

## Dataset

| Dataset | Total | Train | Val | Test |
|---------|-------|-------|-----|------|
| Date-Synth (text images) | 128,510 | 89,957 | 25,702 | 12,851 |
| Products-Synth (product images) | 11,860 | 8,300 | 2,371 | 1,187 |

## Results

**Detection (YOLOv8):**

![Detection Result](https://github.com/HieuNTg/Date-Recognition/assets/96096473/e3e831fa-b112-439f-8f3f-7ad5f7aa7345)

**Text Recognition (CTC-OCR):**

<p align="center">
  <img width="325" alt="OCR Result" src="https://github.com/HieuNTg/Date-Recognition/assets/96096473/0c8f2223-ef85-43cc-a957-2865861b330a">
</p>

## Quick Start

```bash
git clone https://github.com/HieuNTg/Date-Recognition.git
cd Date-Recognition
pip install -r requirements.txt
streamlit run app.py
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
