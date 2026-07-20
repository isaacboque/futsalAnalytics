"""Training script: fine-tune a small YOLO model as a futsal ball detector.

Minimum setup
-------------
1. Export labelled images from Roboflow in YOLOv8 format.
2. Unzip into  data/ball_detector/  (see dataset.yaml).
3. Run::

       python training/ball_detector/train.py [--epochs 100] [--device auto]

The resulting weights land in  runs/ball_detect/train/weights/best.pt.
Pass that path to ``pivotiq analyze --ball-weights ...`` to enable the
fine-tuned detector.

Label count guidance
--------------------
- ~400 annotated source frames → ~1 200 after Roboflow augmentation.
- Aim for ≥ 200 ball instances per scene type.
- Annotate the ball even when partially occluded.
- Include negative frames (no ball visible) at ≈ 20 % of dataset.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_YAML = Path(__file__).parent / "dataset.yaml"
DEFAULT_WEIGHTS = "yolo11n.pt"   # start from the nano checkpoint
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train futsal ball detector")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Starting weights (.pt)")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--device", default="auto")
    p.add_argument("--project", default="runs/ball_detect")
    p.add_argument("--name", default="train")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATASET_YAML}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics not installed. Run: pip install ultralytics") from exc

    model = YOLO(args.weights)
    logger.info(
        "Starting ball detector training: epochs=%d  imgsz=%d  device=%s",
        args.epochs,
        args.imgsz,
        args.device,
    )
    model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    best = Path(args.project) / args.name / "weights" / "best.pt"
    logger.info("Training complete. Best weights: %s", best)
    logger.info(
        "Use with: pivotiq analyze video.mp4 --ball-weights %s", best
    )


if __name__ == "__main__":
    main()
