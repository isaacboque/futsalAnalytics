"""
Entry point for ``python -m futsal_analytics`` and the ``futsal-analytics`` console script.

Supports interactive and headless modes via argparse flags.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

from futsal_analytics.board import TacticalBoard
from futsal_analytics.calibration import (
    FieldCalibrator,
    load_calibration,
    pick_calibration_frame,
    save_calibration,
)
from futsal_analytics.config import Config
from futsal_analytics.detection import (
    BallTracker,
    TeamClassifier,
    make_tracker,
    process_frame,
    resolve_device,
    setup_detectors,
)
from futsal_analytics.field import FieldValidator, SimpleFieldMapper
from futsal_analytics.kpis import KPITracker, PositionLogger
from futsal_analytics.stream import open_youtube_stream, parse_start_time, read_first_frame

logger = logging.getLogger("futsal_analytics")


_SNAPSHOT_WARNED = False


# ---------------------------------------------------------------------------
# Run-lock helpers
# ---------------------------------------------------------------------------
#
# A small ``.run.lock`` file in the output directory prevents two analyser
# processes from clobbering each other's outputs. The active process touches
# the file periodically; readers consider it stale if its mtime is > 10s old.
LOCK_FILENAME = ".run.lock"
LOCK_STALE_AFTER_S = 10.0


def _output_base_dir(args: argparse.Namespace) -> Optional[Path]:
    """Pick a sensible output directory to host the lock file."""
    if getattr(args, "snapshot_dir", None) is not None:
        return Path(args.snapshot_dir)
    for candidate in (args.save_positions, args.save_kpis,
                      args.save_video, args.save_board_video):
        if candidate is not None:
            return Path(candidate).parent
    return None


def _lock_is_active(lock_path: Path) -> bool:
    if not lock_path.exists():
        return False
    try:
        age = time.time() - lock_path.stat().st_mtime
    except OSError:
        return False
    return age < LOCK_STALE_AFTER_S


def _claim_lock(lock_path: Path, url: str) -> bool:
    """Try to claim the lock; return True on success."""
    import os

    if _lock_is_active(lock_path):
        return False
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_path.write_text(
            f"pid={os.getpid()}\nurl={url}\nstart={time.time():.0f}\n",
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Could not write lock file %s: %s", lock_path, exc)
        return False
    return True


def _refresh_lock(lock_path: Path) -> None:
    try:
        lock_path.touch()
    except OSError:
        pass


def _write_snapshot(target: Path, image, max_width: int = 0) -> None:
    """Atomically write an annotated frame to *target* as a JPEG.

    If *max_width* > 0 and the image is wider than that, it is downscaled
    (preserving aspect ratio) before encoding. Cuts JPEG-encode CPU for the
    live preview pipeline by ~75% at 1080p sources.

    Writes to a sibling temp file and renames it, so a Streamlit (or any other)
    reader never sees a half-written file.
    """
    global _SNAPSHOT_WARNED
    try:
        if max_width > 0 and image.shape[1] > max_width:
            scale = max_width / image.shape[1]
            new_h = int(round(image.shape[0] * scale))
            image = cv2.resize(image, (max_width, new_h), interpolation=cv2.INTER_AREA)
        # cv2.imwrite picks the encoder from the file extension, so the tmp
        # file must keep the original suffix (e.g. ``_live_camera.tmp.jpg``).
        tmp = target.with_name(f"{target.stem}.tmp{target.suffix}")
        ok = cv2.imwrite(str(tmp), image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            if not _SNAPSHOT_WARNED:
                logger.warning("Snapshot cv2.imwrite returned False for %s", tmp)
                _SNAPSHOT_WARNED = True
            return
        import os

        os.replace(tmp, target)
    except Exception as exc:
        if not _SNAPSHOT_WARNED:
            logger.warning("Snapshot write failed for %s: %s", target, exc)
            _SNAPSHOT_WARNED = True


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="futsal-analytics",
        description="Real-time futsal match analysis from YouTube streams.",
    )
    p.add_argument(
        "--url",
        help=(
            "YouTube URL or local video file path (prompted if not provided). "
            "Local paths bypass yt-dlp."
        ),
    )
    p.add_argument("--start", default="0", help="Start time MM:SS or seconds (default: 0)")
    p.add_argument(
        "--calibration",
        type=Path,
        help="Path to a saved calibration .npy file (skips the interactive UI)",
    )
    p.add_argument(
        "--save-calibration",
        type=Path,
        help="If interactive calibration is used, save the resulting points here",
    )
    p.add_argument("--save-video", type=Path, help="Path to write annotated camera video (MP4)")
    p.add_argument(
        "--save-board-video", type=Path, help="Path to write tactical board video (MP4)"
    )
    p.add_argument(
        "--save-positions",
        type=Path,
        help="Path to write per-frame player/ball positions (JSONL)",
    )
    p.add_argument(
        "--save-kpis", type=Path, help="Path to write final per-player KPIs (CSV)"
    )
    p.add_argument(
        "--no-gui",
        action="store_true",
        help="Headless mode: no preview windows (still requires --calibration)",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device for YOLO inference (default: auto)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = unlimited)",
    )
    p.add_argument(
        "--retrain-every",
        type=int,
        default=0,
        help="Re-train team classifier every N frames to adapt to lighting (0 = never)",
    )
    p.add_argument(
        "--skip-when-behind",
        type=float,
        default=0.0,
        help=(
            "If processing falls behind by more than this many seconds (live streams), "
            "skip ahead by reading and discarding frames (0 = disabled)"
        ),
    )
    p.add_argument(
        "--allow-frame-pick",
        action="store_true",
        help="Before interactive calibration, let the user advance frames to pick a good one",
    )
    p.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help=(
            "Write annotated camera + tactical board JPEGs every N frames "
            "(used by the web UI for live previews; 0 = disabled)"
        ),
    )
    p.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Directory to write live snapshots into (defaults to the same parent as --save-positions)",
    )
    p.add_argument(
        "--snapshot-width",
        type=int,
        default=960,
        help=(
            "Downscale live snapshots to this width before encoding (0 = native "
            "resolution). Smaller previews mean less I/O and faster browser refresh."
        ),
    )
    return p


# ---------------------------------------------------------------------------
# Interactive prompts (used when flags are missing and stdin is a TTY)
# ---------------------------------------------------------------------------


def show_welcome() -> None:
    print()
    print("+" + "=" * 68 + "+")
    print("|" + "FUTSAL ANALYTICS — REAL-TIME TEAM ANALYSIS".center(68) + "|")
    print("+" + "=" * 68 + "+")
    print()


def prompt_url() -> Optional[str]:
    while True:
        print("VIDEO URL")
        print("-" * 70)
        print("Examples:")
        print("  https://www.youtube.com/watch?v=VIDEO_ID")
        print("  https://www.youtube.com/live/STREAM_ID")
        url = input("Enter YouTube URL (or 'q' to quit): ").strip()
        if url.lower() == "q":
            return None
        if not url:
            print("URL cannot be empty.\n")
            continue
        if "youtube.com" not in url and "youtu.be" not in url:
            print("Not a valid YouTube URL.\n")
            continue
        return url


def prompt_start_time() -> str:
    print("START TIME (optional, MM:SS, ENTER to skip)")
    return input("Start time: ").strip()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace, cfg: Optional[Config] = None) -> int:
    """Run the full analysis pipeline. Returns 0 on success."""
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    cfg = cfg or Config()
    show_welcome()

    # ---------- inputs ----------
    url = args.url
    if not url:
        url = prompt_url()
        if url is None:
            logger.info("User cancelled")
            return 0

    start_time_str = args.start
    if start_time_str == "0" and args.url is None:
        start_time_str = prompt_start_time() or "0"

    if args.no_gui and args.calibration is None:
        logger.error("--no-gui requires --calibration <path> (no UI to calibrate with)")
        return 2

    # ---------- stream ----------
    logger.info("Opening YouTube stream...")
    cap = open_youtube_stream(url, cfg)
    if cap is None or not cap.isOpened():
        logger.error("Failed to open stream. Try: pip install --upgrade yt-dlp")
        return 3

    start_seconds = parse_start_time(start_time_str)
    if start_seconds > 0:
        logger.info("Seeking to %ds (best effort — may be ignored on live streams)", start_seconds)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    logger.info("Stream FPS: %.1f", fps)

    # ---------- calibration ----------
    if args.calibration is not None:
        try:
            field_rect = load_calibration(args.calibration)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Calibration load failed: %s", exc)
            cap.release()
            return 4
        first_frame = read_first_frame(cap, cfg)
        if first_frame is None:
            logger.error("Could not read first frame")
            cap.release()
            return 5
    else:
        if args.allow_frame_pick:
            first_frame = pick_calibration_frame(cap)
        else:
            first_frame = read_first_frame(cap, cfg)
        if first_frame is None:
            logger.error("Could not read calibration frame")
            cap.release()
            return 5

        logger.info("Opening calibration UI...")
        calibrator = FieldCalibrator(first_frame.copy())
        field_rect = calibrator.calibrate()

        if len(field_rect) < 4:
            logger.error("Calibration returned only %d points (need >= 4)", len(field_rect))
            cap.release()
            return 6

        if args.save_calibration:
            save_calibration(field_rect, args.save_calibration)

    # ---------- components ----------
    try:
        field_validator = FieldValidator(field_rect)
        mapper = SimpleFieldMapper(field_rect, cfg.board_width, cfg.board_height)
    except Exception as exc:
        logger.error("Setup failed: %s", exc)
        cap.release()
        return 7

    board_drawer = TacticalBoard(cfg.board_width, cfg.board_height)
    classifier = TeamClassifier()
    ball_tracker = BallTracker(max_distance=150)
    tracker = make_tracker()
    kpi_tracker = KPITracker(fps=fps, board_width_px=cfg.board_width, board_height_px=cfg.board_height)

    # ---------- output base + run-lock ----------
    # Claim the lock BEFORE any other I/O resources are created. Every exit
    # path from this point on goes through the shared try/finally below.
    output_base = _output_base_dir(args) or Path("out")
    lock_path: Optional[Path] = None
    if any([args.save_positions, args.save_kpis, args.save_video, args.save_board_video]):
        lock_path = output_base / LOCK_FILENAME
        if not _claim_lock(lock_path, url):
            try:
                content = lock_path.read_text(encoding="utf-8")
            except OSError:
                content = "(unreadable)"
            logger.error(
                "Another analyser appears to be running in %s.\nLock contents:\n%s\n"
                "If you're sure no other run is active, delete %s and try again.",
                output_base, content, lock_path,
            )
            cap.release()
            return 9
        logger.info("Claimed run lock at %s", lock_path)

    # Declared up here so the finally block can release them whatever happens.
    position_logger: Optional[PositionLogger] = None
    cam_writer: Optional[cv2.VideoWriter] = None
    board_writer: Optional[cv2.VideoWriter] = None
    snapshot_dir: Optional[Path] = None
    frame_idx = 0
    t_start = time.time()

    try:
        if args.save_positions:
            position_logger = PositionLogger(args.save_positions)

        device = resolve_device(args.device)
        logger.info("YOLO device: %s", device)
        model = setup_detectors(cfg)
        if model is None:
            return 8

        # ---------- video writers ----------
        if args.save_video:
            args.save_video.parent.mkdir(parents=True, exist_ok=True)
            fh, fw = first_frame.shape[:2]
            cam_writer = cv2.VideoWriter(
                str(args.save_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (fw, fh),
            )
            logger.info("Recording annotated camera video → %s", args.save_video)
        if args.save_board_video:
            args.save_board_video.parent.mkdir(parents=True, exist_ok=True)
            board_writer = cv2.VideoWriter(
                str(args.save_board_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (cfg.board_width, cfg.board_height),
            )
            logger.info("Recording tactical board video → %s", args.save_board_video)

        # ---------- live snapshot directory ----------
        if args.snapshot_every > 0:
            snapshot_dir = args.snapshot_dir or output_base
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            for stale in ("_live_camera.jpg", "_live_board.jpg"):
                (snapshot_dir / stale).unlink(missing_ok=True)
            logger.info("Live snapshots every %d frames \u2192 %s", args.snapshot_every, snapshot_dir)

        # ---------- preview windows ----------
        if not args.no_gui:
            cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tactical Board", cv2.WINDOW_NORMAL)

        logger.info("=" * 70)
        logger.info("ANALYSIS STARTED — press Q in a window to stop (Ctrl+C in headless mode)")
        logger.info("=" * 70)

        # Process the initial frame too, since we already read it
        current_frame = first_frame
        while True:
            if current_frame is None:
                ret, current_frame = cap.read()
                if not ret or current_frame is None:
                    logger.info("End of stream reached")
                    break

            frame_idx += 1
            if frame_idx % 30 == 0:
                logger.info(
                    "Frame %d — wall %.1fs, video %.1fs",
                    frame_idx,
                    time.time() - t_start,
                    frame_idx / fps,
                )

            # Live-stream catch-up: drop frames if behind by more than threshold
            if args.skip_when_behind > 0:
                wall_elapsed = time.time() - t_start
                video_elapsed = frame_idx / fps
                lag = wall_elapsed - video_elapsed
                if lag > args.skip_when_behind:
                    skip_n = int(lag * fps)
                    logger.warning("Behind by %.1fs — skipping %d frames", lag, skip_n)
                    for _ in range(skip_n):
                        ret, _ = cap.read()
                        if not ret:
                            break
                        frame_idx += 1

            try:
                annotated, board_img, players_frame, mapped_ball = process_frame(
                    current_frame,
                    model,
                    classifier,
                    mapper,
                    field_validator,
                    board_drawer,
                    ball_tracker,
                    cfg,
                    fps,
                    tracker=tracker,
                    device=device,
                    frame_idx=frame_idx,
                    retrain_every=args.retrain_every,
                )
            except Exception as exc:
                logger.error("Frame %d error: %s", frame_idx, exc)
                current_frame = None
                continue

            # KPIs + position log
            kpi_tracker.update(frame_idx - 1, players_frame, mapped_ball)
            if position_logger is not None:
                position_logger.log(frame_idx - 1, (frame_idx - 1) / fps, players_frame, mapped_ball)

            # Overlay status
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

            if cam_writer is not None:
                cam_writer.write(annotated)
            if board_writer is not None:
                board_writer.write(board_img)

            if snapshot_dir is not None and frame_idx % args.snapshot_every == 0:
                _write_snapshot(snapshot_dir / "_live_camera.jpg", annotated, args.snapshot_width)
                _write_snapshot(snapshot_dir / "_live_board.jpg", board_img, args.snapshot_width)

            if lock_path is not None and frame_idx % 30 == 0:
                _refresh_lock(lock_path)
                # Flush the KPI CSV alongside the lock refresh so the Viewer
                # can show a progressively-updating Overview tab while the run
                # is still in progress. Atomic write; ~50 ms even with 30 players.
                if args.save_kpis is not None:
                    kpi_tracker.save_csv(args.save_kpis, quiet=True)

            if not args.no_gui:
                cv2.imshow("Camera View", annotated)
                cv2.imshow("Tactical Board", board_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Stopped by user")
                    break

            if args.max_frames and frame_idx >= args.max_frames:
                logger.info("Reached --max-frames=%d", args.max_frames)
                break

            current_frame = None  # force next iteration to read a new one

    finally:
        if lock_path is not None:
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
        cap.release()
        if not args.no_gui:
            cv2.destroyAllWindows()
        if cam_writer is not None:
            cam_writer.release()
        if board_writer is not None:
            board_writer.release()
        if position_logger is not None:
            position_logger.close()

        elapsed = time.time() - t_start
        fps_avg = frame_idx / elapsed if elapsed > 0 else 0.0

        if args.save_kpis:
            kpi_tracker.save_csv(args.save_kpis)

        print()
        print("+" + "=" * 68 + "+")
        print("|" + "ANALYSIS COMPLETE".center(68) + "|")
        print("|" + f"  Frames processed : {frame_idx}".ljust(68) + "|")
        print("|" + f"  Total time       : {elapsed:.1f}s".ljust(68) + "|")
        print("|" + f"  Average FPS      : {fps_avg:.1f}".ljust(68) + "|")
        print("+" + "=" * 68 + "+")
        print()

        if kpi_tracker.stats:
            print("PER-PLAYER KPIs")
            print("-" * 70)
            print(kpi_tracker.summary())
            print()

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception:
        logger.exception("Unexpected error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
