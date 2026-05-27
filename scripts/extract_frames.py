"""
Extract frames from YouTube videos for YOLO dataset creation.

Usage:
    python scripts/extract_frames.py \
        --url "https://www.youtube.com/watch?v=..." \
        --output futsal_dataset/images/train \
        --fps 2 \
        --skip-frames 5 \
        --max-frames 500

This script:
1. Downloads video from YouTube (using yt-dlp)
2. Extracts frames at specified FPS
3. Skips frames to reduce redundancy
4. Saves to output directory with sequential numbering
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_ffmpeg() -> bool:
    """Check if ffmpeg is installed."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error(
            "ffmpeg not found. Install it:\n"
            "  Windows: choco install ffmpeg  (or download from ffmpeg.org)\n"
            "  Mac: brew install ffmpeg\n"
            "  Linux: sudo apt-get install ffmpeg"
        )
        return False


def check_yt_dlp() -> bool:
    """Check if yt-dlp is installed."""
    try:
        import yt_dlp
        return True
    except ImportError:
        logger.error("yt-dlp not found. Install with: pip install yt-dlp")
        return False


def download_youtube_video(url: str, output_path: Path) -> Optional[Path]:
    """
    Download video from YouTube.
    
    Args:
        url: YouTube URL
        output_path: Where to save video
    
    Returns:
        Path to downloaded video, or None if failed
    """
    try:
        import yt_dlp
        
        logger.info(f"Downloading: {url}")
        
        output_template = str(output_path / "%(title)s.%(ext)s")
        ydl_opts = {
            "format": "best[ext=mp4]/best",
            "outtmpl": output_template,
            "quiet": False,
            "no_warnings": False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            video_path = Path(filename)
            
            logger.info(f"✓ Downloaded: {video_path}")
            return video_path
            
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float = 2.0,
    skip_frames: int = 1,
    max_frames: Optional[int] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> int:
    """
    Extract frames from video using ffmpeg.
    
    Args:
        video_path: Path to video file
        output_dir: Where to save frames
        fps: Frames per second to extract
        skip_frames: Extract every Nth frame (after fps filtering)
        max_frames: Stop after this many frames
        start_time: Start timestamp (e.g., "00:01:30")
        end_time: End timestamp (e.g., "00:05:00")
    
    Returns:
        Number of frames extracted
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return 0
    
    # Build ffmpeg command
    cmd = ["ffmpeg", "-i", str(video_path)]
    
    if start_time:
        cmd.extend(["-ss", start_time])
    if end_time:
        cmd.extend(["-to", end_time])
    
    # Filter to extract fps, then skip frames
    filter_str = f"fps={fps}"
    if skip_frames > 1:
        filter_str += f",select='isnan(prev_selected_t)+gte(t\\,prev_selected_t+{skip_frames/fps})'"
    
    cmd.extend([
        "-vf", filter_str,
        "-vsync", "0",
        str(output_dir / "frame_%05d.jpg"),
        "-y",  # Overwrite output files
    ])
    
    logger.info(f"Extracting frames from: {video_path}")
    logger.info(f"FPS: {fps}, Skip frames: {skip_frames}")
    if start_time:
        logger.info(f"Start time: {start_time}")
    if end_time:
        logger.info(f"End time: {end_time}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Count extracted frames
        frame_files = list(output_dir.glob("frame_*.jpg"))
        count = len(frame_files)
        
        if max_frames and count > max_frames:
            logger.info(f"Limiting to {max_frames} frames (extracted {count})")
            for f in sorted(frame_files)[max_frames:]:
                f.unlink()
            count = max_frames
        
        logger.info(f"✓ Extracted {count} frames to: {output_dir}")
        return count
        
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed: {e.stderr}")
        return 0


def create_annotation_yaml(output_dir: Path, dataset_name: str = "futsal_dataset"):
    """Create YOLO dataset config YAML."""
    dataset_root = output_dir.parent.parent  # Go up from images/train
    
    yaml_content = f"""# {dataset_name} - Auto-generated YOLO config
path: {dataset_root.absolute()}
train: images/train
val: images/val
test: images/test

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
        description="Extract frames from YouTube for YOLO dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 2 fps from full video
  python scripts/extract_frames.py \\
    --url "https://www.youtube.com/watch?v=..." \\
    --output futsal_dataset/images/train

  # Extract from specific time range (skip download)
  python scripts/extract_frames.py \\
    --video my_video.mp4 \\
    --output futsal_dataset/images/train \\
    --start 00:05:00 \\
    --end 00:10:00

  # Extract with higher fps, skip redundant frames
  python scripts/extract_frames.py \\
    --url "https://www.youtube.com/watch?v=..." \\
    --output futsal_dataset/images/train \\
    --fps 5 \\
    --skip-frames 3 \\
    --max-frames 1000

  # Split into train/val after extraction
  # See: scripts/split_dataset.py
        """
    )
    
    # Input: YouTube URL or local video
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--url",
        help="YouTube URL"
    )
    input_group.add_argument(
        "--video",
        help="Local video file (skips download)"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for frames (e.g., futsal_dataset/images/train)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second to extract (default: 2.0)"
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Extract every Nth frame (default: 1 = all)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to extract (default: unlimited)"
    )
    parser.add_argument(
        "--start",
        help="Start time (e.g., 00:01:30)"
    )
    parser.add_argument(
        "--end",
        help="End time (e.g., 00:10:00)"
    )
    parser.add_argument(
        "--keep-video",
        action="store_true",
        help="Keep downloaded video after extraction"
    )
    parser.add_argument(
        "--temp-dir",
        default="/tmp" if sys.platform != "win32" else None,
        help="Temporary directory for video download"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_ffmpeg():
        sys.exit(1)
    if args.url and not check_yt_dlp():
        sys.exit(1)
    
    output_dir = Path(args.output)
    
    video_path = None
    
    # Download from YouTube if URL provided
    if args.url:
        temp_dir = Path(args.temp_dir or output_dir.parent)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = download_youtube_video(args.url, temp_dir)
        if not video_path:
            sys.exit(1)
    else:
        video_path = Path(args.video)
    
    # Extract frames
    count = extract_frames(
        video_path=video_path,
        output_dir=output_dir,
        fps=args.fps,
        skip_frames=args.skip_frames,
        max_frames=args.max_frames,
        start_time=args.start,
        end_time=args.end,
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
    logger.info("   Tools: Roboflow, LabelImg, CVAT")
    logger.info("   Format: YOLO (.txt files in labels/ directory)")
    logger.info("")
    logger.info(f"2. Create validation set:")
    logger.info(f"   python scripts/split_dataset.py \\")
    logger.info(f"     --dataset futsal_dataset")
    logger.info("")
    logger.info(f"3. Train YOLO:")
    logger.info(f"   python scripts/train.py \\")
    logger.info(f"     --data futsal_dataset.yaml")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
