"""
YOLO11 Training Script for Futsal Analytics

Trains a YOLO model on futsal-specific data with support for GPU/CPU training,
resume capabilities, and validation metrics.

Usage:
    python scripts/train.py --data futsal_data.yaml --model yolo11n.pt --epochs 50
    python scripts/train.py --data futsal_data.yaml --resume  # Resume last training
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """Check if ultralytics is installed."""
    try:
        import ultralytics
        logger.info(f"ultralytics version: {ultralytics.__version__}")
        return True
    except ImportError:
        logger.error("ultralytics not found. Install with: pip install ultralytics")
        return False


def check_gpu() -> bool:
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            return True
        else:
            logger.warning("No GPU detected. Training will be CPU-only (slow).")
            return False
    except Exception as e:
        logger.warning(f"Could not check GPU: {e}")
        return False


def validate_dataset_yaml(yaml_path: Path) -> bool:
    """Validate YOLO dataset YAML structure."""
    if not yaml_path.exists():
        logger.error(f"Dataset YAML not found: {yaml_path}")
        return False
    
    try:
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        
        required_keys = ["path", "train", "val"]
        missing = [k for k in required_keys if k not in data]
        
        if missing:
            logger.error(f"Missing required keys in YAML: {missing}")
            return False
        
        # Check if train/val paths exist
        base_path = Path(data["path"])
        train_path = base_path / data["train"]
        val_path = base_path / data["val"]
        
        if not train_path.exists():
            logger.error(f"Train path not found: {train_path}")
            return False
        if not val_path.exists():
            logger.warning(f"Val path not found: {val_path} (will skip validation)")
        
        logger.info(f"✓ Dataset structure valid")
        return True
    except Exception as e:
        logger.error(f"Error validating YAML: {e}")
        return False


def train(
    data: str,
    model: str = "yolo11n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
    device: Optional[str] = None,
    patience: int = 10,
    resume: bool = False,
    project: str = "runs/detect",
    name: str = "futsal_train",
    save_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Train YOLO model on futsal dataset.
    
    Args:
        data: Path to dataset YAML file
        model: Model to train (yolo11n, yolo11s, yolo11m, etc.)
        epochs: Number of training epochs
        imgsz: Image size (should be multiple of 32)
        batch_size: Batch size (reduce if VRAM limited)
        device: Device ID (0 for first GPU, 'cpu' for CPU only)
        patience: Early stopping patience
        resume: Resume from last checkpoint
        project: Project directory
        name: Experiment name
        save_dir: Optional output directory override
    
    Returns:
        Path to best model weights, or None if training failed
    """
    
    if not check_dependencies():
        return None
    
    # Validate dataset
    data_path = Path(data)
    if not validate_dataset_yaml(data_path):
        return None
    
    # Check GPU
    has_gpu = check_gpu()
    if device is None:
        device = 0 if has_gpu else "cpu"
    elif device == "auto":
        device = 0 if has_gpu else "cpu"
    
    # Batch size warning for limited VRAM
    if device == "cpu":
        if batch_size > 8:
            logger.warning("Reducing batch size to 4 for CPU training")
            batch_size = 4
    elif isinstance(device, int):
        # GPU available
        if batch_size > 32:
            logger.warning("Large batch size on GPU. Monitor VRAM usage.")
    
    try:
        from ultralytics import YOLO
        
        logger.info(f"Loading model: {model}")
        model_obj = YOLO(model)
        
        logger.info("="*60)
        logger.info("TRAINING CONFIGURATION")
        logger.info("="*60)
        logger.info(f"Dataset: {data_path.absolute()}")
        logger.info(f"Model: {model}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Image size: {imgsz}x{imgsz}")
        logger.info(f"Device: {device}")
        logger.info(f"Early stopping patience: {patience}")
        logger.info(f"Resume: {resume}")
        logger.info("="*60)
        
        # Train
        results = model_obj.train(
            data=str(data_path.absolute()),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            patience=patience,
            project=project,
            name=name,
            resume=resume,
            
            # Optimization
            close_mosaic=10,  # Stop mosaic augmentation near end
            mosaic=1.0,       # 100% mosaic augmentation (good for small objects like ball)
            workers=0,        # Disable multiprocessing to avoid pagefile issues on Windows
            
            # Regularization
            dropout=0.0,
            
            # Validation
            val=True,
            save=True,
            save_period=-1,   # Save only best
            
            # Logging
            verbose=True,
            plots=True,
        )
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        
        # Results summary
        if hasattr(results, 'best_fitness'):
            logger.info(f"Best fitness: {results.best_fitness:.4f}")
        
        # Find the best model - ultralytics creates nested project structure
        # Try the results save_dir first (most reliable)
        if hasattr(results, 'save_dir'):
            best_model_path = Path(results.save_dir) / "weights" / "best.pt"
            if best_model_path.exists():
                logger.info(f"✓ Best model saved: {best_model_path.absolute()}")
                return best_model_path
        
        # Fallback: search for latest trained model in project directory
        project_path = Path(project)
        if project_path.exists():
            # Find all best.pt files and get the latest one
            best_models = list(project_path.glob("**/weights/best.pt"))
            if best_models:
                # Sort by modification time and get the newest
                latest_model = max(best_models, key=lambda p: p.stat().st_mtime)
                logger.info(f"✓ Best model found: {latest_model.absolute()}")
                return latest_model
        
        logger.warning(f"Could not find best model in {project_path}")
        return None
            
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return None


def export_model(weights_path: Path, export_format: str = "onnx") -> Optional[Path]:
    """
    Export trained model to other formats.
    
    Args:
        weights_path: Path to best.pt
        export_format: Format ('onnx', 'tflite', 'torchscript', etc.)
    
    Returns:
        Path to exported model
    """
    try:
        from ultralytics import YOLO
        
        logger.info(f"Exporting model to {export_format}...")
        model = YOLO(str(weights_path))
        
        exported_path = model.export(format=export_format)
        logger.info(f"✓ Model exported: {exported_path}")
        return Path(exported_path)
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO11 on futsal dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training (nano model, 50 epochs)
  python scripts/train.py --data futsal_data.yaml

  # Larger model with GPU
  python scripts/train.py --data futsal_data.yaml --model yolo11m.pt --batch 32

  # Resume previous training
  python scripts/train.py --data futsal_data.yaml --resume

  # CPU only (slow, good for testing)
  python scripts/train.py --data futsal_data.yaml --device cpu --batch 4
        """
    )
    
    parser.add_argument(
        "--data",
        required=True,
        help="Path to dataset YAML file (required)"
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
        help="Model size (default: yolo11n.pt)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs (default: 50)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        dest="batch_size",
        help="Batch size (default: 16, reduce for limited VRAM)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640, must be multiple of 32)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: 0 (GPU), 'cpu' (CPU only), 'auto' (auto-detect, default)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--project",
        default="runs/detect",
        help="Project directory (default: runs/detect)"
    )
    parser.add_argument(
        "--name",
        default="futsal_train",
        help="Experiment name (default: futsal_train)"
    )
    parser.add_argument(
        "--export",
        help="Export model format after training (onnx, tflite, etc.)"
    )
    
    args = parser.parse_args()
    
    # Convert device string to appropriate type
    device = args.device
    if device != "auto" and device != "cpu":
        try:
            device = int(device)
        except ValueError:
            logger.error(f"Invalid device: {device}")
            sys.exit(1)
    
    # Train
    best_model = train(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=device,
        patience=args.patience,
        resume=args.resume,
        project=args.project,
        name=args.name,
    )
    
    # Export if requested
    if best_model and args.export:
        export_model(best_model, args.export)
    
    sys.exit(0 if best_model else 1)


if __name__ == "__main__":
    main()
