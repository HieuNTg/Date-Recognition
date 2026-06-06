"""FastAPI service exposing the expiry-date recognition pipeline as a REST API.

Run from the repository root:

    uvicorn api.main:app --host 0.0.0.0 --port 8000

The pipeline (and its models) is loaded once at startup and reused across
requests — see `src/pipeline.py`.
"""

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.pipeline import DatePipeline

app = FastAPI(
    title="DateReg API",
    description="Detect and read product expiry dates from images (YOLOv8 + CTC-OCR).",
    version="1.0.0",
)

pipeline: DatePipeline | None = None


@app.on_event("startup")
def _load_pipeline():
    global pipeline
    pipeline = DatePipeline()


@app.get("/health")
def health():
    """Liveness/readiness probe — reports whether models are loaded."""
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Detect date regions in an uploaded image and return the parsed expiry date.

    Response:
        {
          "date": "2026-06-01",            # best (latest) detected date, or null
          "status": "valid|warning|expired|null",
          "days_remaining": 360,           # signed; negative == expired
          "detections": [
            {"text": "...", "confidence": 0.97, "bbox": [x, y, x1, y1]}
          ]
        }
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid or unsupported image file")

    result = pipeline.run(image)
    return result.to_dict()
