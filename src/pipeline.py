"""End-to-end expiry-date recognition pipeline.

Orchestrates the three stages — YOLOv8 detection, CTC-OCR recognition, and
date parsing — behind a single, UI-agnostic interface. The Streamlit app, the
REST API and the batch CLI all drive inference through this one class, so the
inference logic lives in exactly one place.
"""

from dataclasses import dataclass, field, asdict

from .detection import YOLODetector
from .recognition import OCRRecognizer
from .utils import DateParser


@dataclass
class Detection:
    text: str
    confidence: float
    bbox: tuple  # (x, y, x1, y1)


@dataclass
class DateResult:
    """Structured output of a single inference run."""
    date: str | None = None              # best (latest) recognised date, raw text
    status: str | None = None            # "valid" | "warning" | "expired" | None
    days_remaining: int | None = None    # signed: negative == days past expiry
    detections: list = field(default_factory=list)  # list[Detection]

    def to_dict(self):
        d = asdict(self)
        d["detections"] = [asdict(det) for det in self.detections]
        return d


class DatePipeline:
    """Detect → recognise → parse, in one call.

    Models are loaded once at construction and the pipeline holds no per-request
    state, so a single instance can be reused across many images (and many
    concurrent API requests).
    """

    def __init__(self, config_path="configs/config.yaml"):
        self.detector = YOLODetector(config_path)
        self.recognizer = OCRRecognizer(config_path)
        self.parser = DateParser(config_path)

    def run(self, image) -> DateResult:
        """Run the full pipeline on a single PIL.Image (RGB)."""
        detections = self.detector.predict(image)

        recognised = []
        for det in detections:
            text = self.recognizer.recognize(det["crop"])
            recognised.append(Detection(
                text=text,
                confidence=det["confidence"],
                bbox=det["bbox"],
            ))

        if not recognised:
            return DateResult()

        best_date = self.parser.get_max_date([d.text for d in recognised])
        status, delta = self.parser.evaluate_expiry(best_date) if best_date else (None, None)

        return DateResult(
            date=best_date,
            status=status,
            days_remaining=delta,
            detections=recognised,
        )
