"""Batch expiry-date recognition over a folder of images.

Usage:
    python cli.py path/to/images/ --out results.json
    python cli.py path/to/images/ --out results.csv --format csv

Drives the same `DatePipeline` used by the web app and the API, so results are
identical across all three entry points.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.pipeline import DatePipeline

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(root: Path):
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def main():
    ap = argparse.ArgumentParser(description="Batch expiry-date recognition.")
    ap.add_argument("input", type=Path, help="Image file or a folder of images")
    ap.add_argument("--out", type=Path, default=None, help="Output file (json or csv)")
    ap.add_argument("--format", choices=["json", "csv"], default="json")
    ap.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = ap.parse_args()

    images = collect_images(args.input)
    if not images:
        print(f"No images found under {args.input}", file=sys.stderr)
        sys.exit(1)

    pipeline = DatePipeline(args.config)
    rows = []
    for path in images:
        try:
            image = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            print(f"  skip (unreadable): {path}", file=sys.stderr)
            continue

        result = pipeline.run(image)
        rows.append({
            "file": str(path),
            "date": result.date,
            "status": result.status,
            "days_remaining": result.days_remaining,
            "num_detections": len(result.detections),
        })
        print(f"  {path.name}: {result.date or '—'} [{result.status or 'n/a'}]")

    if args.out:
        if args.format == "csv":
            with open(args.out, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        else:
            args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {len(rows)} results → {args.out}")


if __name__ == "__main__":
    main()
