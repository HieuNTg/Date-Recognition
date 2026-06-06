import yaml
from ultralytics import YOLO


class YOLODetector:
    """Locates date regions on a product image with a fine-tuned YOLOv8 model.

    Crops are returned in-memory (PIL) rather than written to disk, so the
    detector is stateless and safe to share across concurrent requests.
    """

    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["model"]["yolo"]

        self.model = YOLO(cfg["weights"])
        self.imgsz = cfg["imgsz"]
        self.confidence = cfg["confidence"]
        self.padding = cfg["padding"]

    def predict(self, img):
        """Detect date regions.

        Args:
            img: a PIL.Image (RGB).

        Returns:
            list of dicts: {"crop": PIL.Image, "confidence": float, "bbox": (x, y, x1, y1)}
            sorted by detection confidence (highest first).
        """
        w, h = img.size

        result = self.model.predict(img, save=False, imgsz=self.imgsz, conf=self.confidence)
        boxes = result[0].boxes.data.tolist()

        detections = []
        for box in boxes:
            x = max(0, int(box[0]) - self.padding)
            y = max(0, int(box[1]) - self.padding)
            x1 = min(w, int(box[2]) + self.padding)
            y1 = min(h, int(box[3]) + self.padding)
            conf = float(box[4])

            detections.append({
                "crop": img.crop((x, y, x1, y1)),
                "confidence": conf,
                "bbox": (x, y, x1, y1),
            })

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections
