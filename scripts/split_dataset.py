"""
Split extracted frames into train/val/test sets for YOLO training.

Creates symbolic links (or copies) to organize frames into train/val/test
directories without duplicating files.

Usage:
    python scripts/split_dataset.py --dataset futsal_dataset --split 0.7 0.2 0.1
"""

import argparse
import logging
import random
import shutil
from pathlib import Path
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def split_dataset(
    images_dir: Path,
    labels_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[int, int, int]:
    """
    Split images and labels into train/val/test.
    
    Creates subdirectories (train/, val/, test/) and moves files accordingly.
    
    Args:
        images_dir: Directory containing all images
        labels_dir: Directory containing all labels
        train_ratio: Proportion for training (default: 0.7)
        val_ratio: Proportion for validation (default: 0.2)
        test_ratio: Proportion for testing (default: 0.1)
        seed: Random seed for reproducibility
    
    Returns:
        (train_count, val_count, test_count)
    """
    
    if not (abs((train_ratio + val_ratio + test_ratio) - 1.0) < 0.01):
        raise ValueError("Ratios must sum to 1.0")
    
    random.seed(seed)
    
    # Get all image files
    image_files = sorted([f for f in images_dir.glob("*.jpg") if f.is_file()])
    
    if not image_files:
        logger.error(f"No images found in {images_dir}")
        return 0, 0, 0
    
    logger.info(f"Found {len(image_files)} images")
    
    # Shuffle and split
    random.shuffle(image_files)
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # Create directories
    for split_name in ["train", "val", "test"]:
        (images_dir.parent / split_name).mkdir(exist_ok=True)
        (labels_dir.parent / split_name).mkdir(exist_ok=True)
    
    def move_pair(img_file: Path, split: str) -> bool:
        """Move image and corresponding label file."""
        label_file = labels_dir / img_file.stem + ".txt"
        
        if not label_file.exists():
            logger.warning(f"Label not found for {img_file.name}, skipping")
            return False
        
        # Move image
        dest_img = images_dir.parent / split / img_file.name
        shutil.move(str(img_file), str(dest_img))
        
        # Move label
        dest_label = labels_dir.parent / split / label_file.name
        shutil.move(str(label_file), str(dest_label))
        
        return True
    
    # Move files
    logger.info(f"Moving {len(train_files)} to train/")
    train_count = sum(move_pair(f, "train") for f in train_files)
    
    logger.info(f"Moving {len(val_files)} to val/")
    val_count = sum(move_pair(f, "val") for f in val_files)
    
    logger.info(f"Moving {len(test_files)} to test/")
    test_count = sum(move_pair(f, "test") for f in test_files)
    
    return train_count, val_count, test_count


def main():
    parser = argparse.ArgumentParser(
        description="Split YOLO dataset into train/val/test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default 70/20/10 split
  python scripts/split_dataset.py --dataset futsal_dataset

  # Custom split 80/10/10
  python scripts/split_dataset.py --dataset futsal_dataset --split 0.8 0.1 0.1

  # Specify custom directories
  python scripts/split_dataset.py \\
    --images my_images \\
    --labels my_labels \\
    --split 0.7 0.2 0.1
        """
    )
    
    parser.add_argument(
        "--dataset",
        help="Dataset root directory (assumes images/ and labels/ subdirectories)"
    )
    parser.add_argument(
        "--images",
        help="Images directory (if not using --dataset)"
    )
    parser.add_argument(
        "--labels",
        help="Labels directory (if not using --dataset)"
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=[0.7, 0.2, 0.1],
        help="Train/val/test ratios (default: 0.7 0.2 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Determine directories
    if args.dataset:
        images_dir = Path(args.dataset) / "images"
        labels_dir = Path(args.dataset) / "labels"
    elif args.images and args.labels:
        images_dir = Path(args.images)
        labels_dir = Path(args.labels)
    else:
        parser.error("Provide either --dataset or both --images and --labels")
    
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return 1
    
    if not labels_dir.exists():
        logger.warning(f"Labels directory not found: {labels_dir} (creating empty)")
        labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Split
    train, val, test = split_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        train_ratio=args.split[0],
        val_ratio=args.split[1],
        test_ratio=args.split[2],
        seed=args.seed,
    )
    
    logger.info("=" * 60)
    logger.info(f"✓ Split complete:")
    logger.info(f"  Train: {train} pairs")
    logger.info(f"  Val:   {val} pairs")
    logger.info(f"  Test:  {test} pairs")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
