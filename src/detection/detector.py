import os
from pathlib import Path
from ultralytics import YOLO
import yaml


class YOLODetector:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["model"]["yolo"]

        self.model = YOLO(cfg["weights"])
        self.imgsz = cfg["imgsz"]
        self.confidence = cfg["confidence"]
        self.padding = cfg["padding"]
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)

    def predict(self, img):
        w, h = img.size

        result = self.model.predict(img, save=False, imgsz=self.imgsz, conf=self.confidence)
        boxes = result[0].boxes.data.tolist()

        detections = []
        for idx, box in enumerate(boxes, start=1):
            x = max(0, int(box[0]) - self.padding)
            y = max(0, int(box[1]) - self.padding)
            x1 = min(w, int(box[2]) + self.padding)
            y1 = min(h, int(box[3]) + self.padding)
            conf = float(box[4])

            crop = img.crop((x, y, x1, y1))
            filepath = str(self.temp_dir / f"crop_{idx}.jpg")
            crop.save(filepath)

            detections.append({
                "filepath": filepath,
                "confidence": conf,
                "bbox": (x, y, x1, y1),
            })

        return detections

    def cleanup(self):
        for f in self.temp_dir.glob("crop_*.jpg"):
            f.unlink()
