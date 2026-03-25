# DateReg — Expiry Date Recognition System

An end-to-end deep learning pipeline that detects and reads expiry dates from product images, built with **YOLOv8** for detection and a **CTC-based OCR** model for text recognition.

<p align="center">
  <img width="1024" alt="Demo" src="https://github.com/HieuNTg/Date-Recognition/assets/96096473/23da4d31-45d7-4824-96bd-3b2aa25b090f">
</p>

## About

Trong ngành thực phẩm và bán lẻ, việc kiểm tra hạn sử dụng sản phẩm vẫn chủ yếu được thực hiện thủ công — tốn thời gian, dễ sai sót, và khó mở rộng quy mô. **DateReg** giải quyết vấn đề này bằng cách tự động hóa toàn bộ quy trình thông qua Computer Vision:

1. **Phát hiện** — Mô hình YOLOv8 được huấn luyện trên 11,860 ảnh sản phẩm để định vị chính xác vùng chứa ngày tháng trên bao bì
2. **Nhận diện** — Mô hình CRNN (CNN + BiLSTM + CTC) được huấn luyện trên 128,510 ảnh text để đọc ký tự từ vùng date đã crop
3. **Phân tích** — Hệ thống tự động parse nhiều định dạng ngày, xử lý lỗi OCR phổ biến, và đánh giá trạng thái hết hạn

Dự án hướng đến ứng dụng trong quản lý kho hàng, kiểm tra chất lượng tại siêu thị, và hệ thống giám sát sản phẩm tự động.

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

|  | Precision | Recall | mAP50 | mAP50-95 |
|---|:-:|:-:|:-:|:-:|
| **Training** | 0.969 | 0.963 | 0.981 | 0.862 |
| **Test** | 0.960 | 0.963 | 0.976 | 0.874 |

**Text Recognition (CTC-OCR):**

| Metric | Score |
|---|:-:|
| **CER** (Character Error Rate) | 0.05 |
| **WER** (Word Error Rate) | 0.19 |

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
