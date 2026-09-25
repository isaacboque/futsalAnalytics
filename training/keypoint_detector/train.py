"""Training script: fine-tune YOLO-Pose as a futsal pitch keypoint detector.

The model predicts the locations of up to 29 canonical pitch landmarks in a
single video frame, enabling per-frame homography computation.

Minimum setup
-------------
1. Label ≥ 500 frames with visible landmark pixel coordinates (Roboflow Pose).
2. Export in YOLOv8-Pose format and unzip into  data/keypoint_detector/.
3. Run::

       python training/keypoint_detector/train.py [--epochs 150] [--device auto]

The resulting weights land in  runs/kp_detect/train/weights/best.pt.
Pass that path to ``pivotiq analyze --keypoint-weights ...``.

Evaluation
----------
After training, run the evaluation harness::

    python training/keypoint_detector/eval.py --weights runs/kp_detect/train/weights/best.pt

This reports per-keypoint PCK (Percentage of Correct Keypoints) at 10 px and
20 px thresholds, plus the downstream homography reprojection error on the
validation set.

Label count guidance
--------------------
- ~500 source frames minimum; ~1 500 after Roboflow augmentation.
- Include at least 50 frames with no more than 4 visible keypoints (tight
  zoom on one goal) — these are the hardest cases for registration.
- Include wide-angle frames where 20+ landmarks are visible.
- Visibility flag: 2 = visible, 1 = partially occluded, 0 = absent/off-frame.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_YAML = Path(__file__).parent / "dataset.yaml"
DEFAULT_WEIGHTS = "yolo11n-pose.pt"
DEFAULT_EPOCHS = 150
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 8   # pose models are heavier — smaller default batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train pitch keypoint detector")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Starting weights (.pt)")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--device", default="auto")
    p.add_argument("--project", default="runs/kp_detect")
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
        "Starting keypoint detector training: epochs=%d  imgsz=%d  device=%s",
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
        "Use with: pivotiq analyze video.mp4 --keypoint-weights %s", best
    )


if __name__ == "__main__":
    main()
