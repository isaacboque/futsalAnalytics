"""
Extract frames from YouTube videos using OpenCV (no FFmpeg needed).

Works directly with video files using opencv-python.

Usage:
    python scripts/extract_frames_opencv.py \
        --video my_video.mp4 \
        --output futsal_dataset/images/train \
        --fps 2 \
        --max-frames 500

Or download from YouTube first, then extract:
    # Install yt-dlp
    pip install yt-dlp
    
    # Download best MP4 quality
    yt-dlp -f "best[ext=mp4]" https://www.youtube.com/watch?v=VIDEO_ID -o "video.mp4"
    
    # Extract frames
    python scripts/extract_frames_opencv.py --video video.mp4 --output futsal_dataset/images/train
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import subprocess

try:
    import cv2
except ImportError:
    print("OpenCV not found. Install with: pip install opencv-python")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_youtube_video(url: str, output_path: Path) -> Optional[Path]:
    """
    Download video from YouTube using yt-dlp.
    
    Args:
        url: YouTube URL
        output_path: Where to save video
    
    Returns:
        Path to downloaded video, or None if failed
    """
    try:
        import yt_dlp
        
        logger.info(f"Downloading: {url}")
        
        output_template = str(output_path / "video.mp4")
        ydl_opts = {
            "format": "best[ext=mp4]",
            "outtmpl": output_template,
            "quiet": False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            video_path = Path(filename)
            
            logger.info(f"✓ Downloaded: {video_path}")
            return video_path
            
    except ImportError:
        logger.error("yt-dlp not found. Install with: pip install yt-dlp")
        return None
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


def extract_frames_opencv(
    video_path: Path,
    output_dir: Path,
    target_fps: float = 2.0,
    skip_frames: int = 1,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> int:
    """
    Extract frames from video using OpenCV.
    
    Args:
        video_path: Path to video file
        output_dir: Where to save frames
        target_fps: Target frames per second
        skip_frames: Extract every Nth frame
        max_frames: Stop after this many extracted frames
        start_frame: Start from this frame number
        end_frame: Stop at this frame number
    
    Returns:
        Number of frames extracted
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return 0
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return 0
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video: {width}x{height}, {fps:.1f} fps, {total_frames} total frames")
        logger.info(f"Target FPS: {target_fps}, Skip frames: {skip_frames}")
        
        if fps <= 0:
            fps = 30.0  # Default fallback
        
        # Calculate frame interval based on target FPS
        frame_interval = max(1, int(fps / target_fps)) * skip_frames
        
        frame_count = 0
        extracted_count = 0
        output_index = 1
        
        # Set start position
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_count = start_frame
        
        logger.info("Extracting frames...")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Check end condition
            if end_frame and frame_count >= end_frame:
                break
            
            # Extract at intervals
            if frame_count % frame_interval == 0:
                output_filename = output_dir / f"frame_{output_index:05d}.jpg"
                success = cv2.imwrite(str(output_filename), frame)
                
                if success:
                    extracted_count += 1
                    if extracted_count % 50 == 0:
                        logger.info(f"  Extracted {extracted_count} frames...")
                    
                    if max_frames and extracted_count >= max_frames:
                        logger.info(f"Reached max frames limit ({max_frames})")
                        break
                else:
                    logger.warning(f"Failed to write: {output_filename}")
                
                output_index += 1
            
            frame_count += 1
        
        cap.release()
        
        logger.info(f"✓ Extracted {extracted_count} frames to: {output_dir}")
        return extracted_count
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        return 0


def create_annotation_yaml(output_dir: Path, dataset_name: str = "futsal_dataset"):
    """Create YOLO dataset config YAML."""
    dataset_root = output_dir.parent.parent  # Go up from images/train or images/all
    
    yaml_content = f"""# {dataset_name} - Auto-generated YOLO config
path: {dataset_root.absolute()}
train: images/train
val: images/val

nc: 80
names:
  0: person
  32: sports ball
"""
    
    yaml_path = dataset_root / f"{dataset_name}.yaml"
    yaml_path.write_text(yaml_content)
    logger.info(f"✓ Created YOLO config: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from YouTube or local video (OpenCV-based, no FFmpeg needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from local video (fastest)
  python scripts/extract_frames_opencv.py \\
    --video my_video.mp4 \\
    --output futsal_dataset/images/all

  # Download from YouTube and extract
  python scripts/extract_frames_opencv.py \\
    --url "https://www.youtube.com/watch?v=..." \\
    --output futsal_dataset/images/all

  # Higher FPS with frame skipping
  python scripts/extract_frames_opencv.py \\
    --video my_video.mp4 \\
    --output futsal_dataset/images/all \\
    --fps 5 \\
    --skip-frames 2 \\
    --max-frames 500

  # Pre-download with yt-dlp, then extract:
  yt-dlp -f "best[ext=mp4]" https://www.youtube.com/watch?v=... -o "video.mp4"
  python scripts/extract_frames_opencv.py --video video.mp4 --output futsal_dataset/images/all
        """
    )
    
    # Input: YouTube URL or local video
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--url",
        help="YouTube URL (requires yt-dlp)"
    )
    input_group.add_argument(
        "--video",
        help="Local video file"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for frames (e.g., futsal_dataset/images/all)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Target frames per second (default: 2.0)"
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Extract every Nth frame (default: 1)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to extract (default: unlimited)"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start from this frame number (default: 0)"
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Stop at this frame number (default: unlimited)"
    )
    parser.add_argument(
        "--keep-video",
        action="store_true",
        help="Keep downloaded video after extraction"
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Temporary directory for video download"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    video_path = None
    
    # Download from YouTube if URL provided
    if args.url:
        temp_dir = Path(args.temp_dir or output_dir.parent.parent)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = download_youtube_video(args.url, temp_dir)
        if not video_path:
            sys.exit(1)
    else:
        video_path = Path(args.video)
    
    # Extract frames
    count = extract_frames_opencv(
        video_path=video_path,
        output_dir=output_dir,
        target_fps=args.fps,
        skip_frames=args.skip_frames,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    
    if count == 0:
        logger.error("No frames extracted")
        sys.exit(1)
    
    # Create YOLO config
    create_annotation_yaml(output_dir)
    
    # Cleanup video if requested
    if args.url and not args.keep_video and video_path:
        try:
            video_path.unlink()
            logger.info(f"Cleaned up: {video_path}")
        except Exception as e:
            logger.warning(f"Could not delete {video_path}: {e}")
    
    logger.info("=" * 60)
    logger.info("NEXT STEPS:")
    logger.info("=" * 60)
    logger.info(f"1. Annotate frames in: {output_dir}")
    logger.info("   → Roboflow.com (easiest, free)")
    logger.info("   → Or LabelImg, CVAT, etc.")
    logger.info("")
    logger.info(f"2. Split into train/val:")
    logger.info(f"   python scripts/split_dataset.py --dataset {output_dir.parent.parent}")
    logger.info("")
    logger.info(f"3. Train YOLO:")
    logger.info(f"   python scripts/train.py --data {output_dir.parent.parent}/futsal_dataset.yaml")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
