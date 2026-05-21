"""
Entry point for ``python -m futsal_analytics`` and the ``futsal-analytics`` console script.
"""

import logging
import time
from typing import Optional

import cv2

from futsal_analytics.board import TacticalBoard
from futsal_analytics.calibration import FieldCalibrator
from futsal_analytics.config import Config
from futsal_analytics.detection import BallTracker, TeamClassifier, process_frame, setup_detectors
from futsal_analytics.field import FieldValidator, SimpleFieldMapper
from futsal_analytics.stream import open_youtube_stream, parse_start_time, read_first_frame

logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# User interface helpers
# ---------------------------------------------------------------------------


def show_welcome() -> None:
    print("\n")
    print("+" + "=" * 68 + "+")
    print("|" + " " * 68 + "|")
    print("|" + "FUTSAL ANALYTICS — REAL-TIME TEAM ANALYSIS".center(68) + "|")
    print("|" + " " * 68 + "|")
    print("+" + "=" * 68 + "+")
    print("\n")


def get_video_url() -> Optional[str]:
    """Prompt the user for a YouTube URL; return None if they quit."""
    while True:
        print("VIDEO URL")
        print("-" * 70)
        print("Examples:")
        print("  https://www.youtube.com/watch?v=VIDEO_ID")
        print("  https://www.youtube.com/live/STREAM_ID")
        print()
        url = input("Enter YouTube URL (or 'q' to quit): ").strip()

        if url.lower() == "q":
            return None
        if not url:
            print("URL cannot be empty. Try again.\n")
            continue
        if "youtube.com" not in url and "youtu.be" not in url:
            print("Not a valid YouTube URL. Try again.\n")
            continue

        print(f"URL accepted: {url[:60]}\n")
        return url


def get_start_time() -> str:
    """Prompt for an optional start time; return empty string to start from beginning."""
    print("START TIME (optional)")
    print("-" * 70)
    print("Format: MM:SS  e.g. 02:30 starts at 2 minutes 30 seconds")
    print("Press ENTER to start from the beginning")
    print()
    time_str = input("Start time: ").strip()
    if time_str:
        print(f"Starting from: {time_str}\n")
    else:
        print("Starting from the beginning\n")
    return time_str


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(cfg: Optional[Config] = None) -> None:
    """Full analysis pipeline."""
    cfg = cfg or Config()

    show_welcome()
    logger.info("Config: %s", cfg)

    url = get_video_url()
    if url is None:
        logger.info("User cancelled")
        return

    start_time_str = get_start_time()

    logger.info("[STREAM] Opening YouTube stream...")
    cap = open_youtube_stream(url, cfg)
    if cap is None or not cap.isOpened():
        logger.error("[ERROR] Failed to open stream")
        logger.error("  • pip install --upgrade yt-dlp")
        logger.error("  • Verify the video is public and not age-restricted")
        return

    logger.info("[OK] Stream opened")

    start_seconds = parse_start_time(start_time_str)
    if start_seconds > 0:
        logger.info("[SEEK] Seeking to %ds...", start_seconds)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    logger.info("[VIDEO] FPS: %.1f", fps)

    logger.info("[FRAME] Reading first frame...")
    first_frame = read_first_frame(cap, cfg)
    if first_frame is None:
        logger.error("[ERROR] Could not read first frame")
        cap.release()
        return

    logger.info("[FRAME] Dimensions: %dx%d", first_frame.shape[1], first_frame.shape[0])

    logger.info("[CALIBRATION] Opening calibration UI...")
    calibrator = FieldCalibrator(first_frame.copy())
    field_rect = calibrator.calibrate()

    if len(field_rect) < 4:
        logger.error("[ERROR] Calibration returned only %d points (need >= 4)", len(field_rect))
        cap.release()
        return

    logger.info("[CALIBRATION] Completed with %d points", len(field_rect))

    try:
        field_validator = FieldValidator(field_rect)
        mapper = SimpleFieldMapper(field_rect, cfg.board_width, cfg.board_height)
    except Exception as exc:
        logger.error("[ERROR] Setup failed: %s", exc)
        cap.release()
        return

    board_drawer = TacticalBoard(cfg.board_width, cfg.board_height)
    classifier = TeamClassifier()
    ball_tracker = BallTracker(max_distance=150)

    logger.info("[YOLO] Loading detector...")
    model = setup_detectors(cfg)
    if model is None:
        logger.error("[ERROR] Failed to load YOLO model")
        cap.release()
        return

    cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Tactical Board", cv2.WINDOW_NORMAL)

    frame_idx = 0
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("REAL-TIME ANALYSIS STARTED — press Q to stop")
    logger.info("=" * 70)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info("[STREAM] End of stream reached")
                break

            frame_idx += 1
            if frame_idx % 10 == 0:
                logger.info("[PROGRESS] Frame %d — %.1fs elapsed", frame_idx, time.time() - t_start)

            try:
                annotated, board_img = process_frame(
                    frame, model, classifier, mapper,
                    field_validator, board_drawer, ball_tracker, cfg, fps,
                )
            except Exception as exc:
                logger.error("[ERROR] Frame %d: %s", frame_idx, exc)
                continue

            elapsed = time.time() - t_start
            cv2.putText(
                annotated,
                f"Frame {frame_idx} | {elapsed:.1f}s",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Camera View", annotated)
            cv2.imshow("Tactical Board", board_img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("[USER] Analysis stopped by user")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        fps_avg = frame_idx / elapsed if elapsed > 0 else 0.0

        print("\n")
        print("+" + "=" * 68 + "+")
        print("|" + "ANALYSIS COMPLETE".center(68) + "|")
        print("|" + " " * 68 + "|")
        print("|" + f"  Frames processed : {frame_idx}".ljust(68) + "|")
        print("|" + f"  Total time       : {elapsed:.1f}s".ljust(68) + "|")
        print("|" + f"  Average FPS      : {fps_avg:.1f}".ljust(68) + "|")
        print("|" + " " * 68 + "|")
        print("+" + "=" * 68 + "+")
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
