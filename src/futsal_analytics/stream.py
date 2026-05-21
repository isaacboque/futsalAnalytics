"""YouTube stream opening and frame-reading utilities."""

import logging
import subprocess
import time
from typing import Optional

import cv2

from futsal_analytics.config import Config

logger = logging.getLogger(__name__)


def open_youtube_stream(url: str, config: Config) -> Optional[cv2.VideoCapture]:
    """
    Open a YouTube video — or a local file path — as an OpenCV VideoCapture.

    If *url* is a local file path that exists on disk, it is opened directly
    by OpenCV. Otherwise it is resolved via yt-dlp, trying resolutions in
    descending order (1080p → 720p → 480p → 360p → best).

    Args:
        url: Public YouTube URL, live-stream URL, or local video file path.
        config: System configuration (uses ``yt_dlp_timeout``).

    Returns:
        An opened ``cv2.VideoCapture``, or ``None`` on failure.
    """
    # Local-file fast path — no yt-dlp needed
    from pathlib import Path

    local_candidate = Path(url)
    if local_candidate.exists() and local_candidate.is_file():
        cap = cv2.VideoCapture(str(local_candidate))
        if cap.isOpened():
            logger.info("Opened local file: %s", local_candidate)
            return cap
        cap.release()
        logger.error("OpenCV could not open local file: %s", local_candidate)
        return None

    formats = [
        "best[height<=1080]",
        "best[height<=720]",
        "best[height<=480]",
        "best[height<=360]",
        "best",
    ]

    for fmt in formats:
        try:
            result = subprocess.run(
                ["yt-dlp", "-f", fmt, "--get-url", url],
                capture_output=True,
                text=True,
                timeout=config.yt_dlp_timeout,
            )

            if result.returncode != 0:
                logger.warning("yt-dlp failed for %s: %s", fmt, result.stderr.strip()[:120])
                continue

            stream_url = result.stdout.strip().split("\n")[0]
            if not stream_url:
                continue

            cap = cv2.VideoCapture(stream_url)
            if cap.isOpened():
                logger.info("Stream opened successfully (%s)", fmt)
                return cap
            cap.release()

        except FileNotFoundError:
            logger.error("yt-dlp not found. Install with: pip install yt-dlp")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("yt-dlp timed out for %s", fmt)

    logger.error("Failed to open stream with any format")
    return None


def read_first_frame(cap: cv2.VideoCapture, config: Config) -> Optional[cv2.typing.MatLike]:
    """
    Read the first valid frame from a VideoCapture, retrying on empty reads.

    Args:
        cap: An open VideoCapture.
        config: System configuration (uses ``stream_read_retries`` and
                ``stream_read_delay``).

    Returns:
        The first frame as a NumPy array, or ``None`` if all retries fail.
    """
    for attempt in range(config.stream_read_retries):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            logger.info("First frame obtained after %d attempt(s)", attempt)
            return frame

        if attempt < config.stream_read_retries - 1:
            logger.debug(
                "Waiting for stream buffer... (%d/%d)",
                attempt + 1,
                config.stream_read_retries,
            )
            time.sleep(config.stream_read_delay)

    logger.error("Could not read the first frame")
    return None


def parse_start_time(time_str: str) -> int:
    """
    Parse a ``HH:MM:SS``, ``MM:SS`` or ``SS`` time string into total seconds.

    Returns 0 on invalid input (with a warning logged).
    """
    if time_str is None:
        return 0
    try:
        parts = [int(p) for p in time_str.strip().split(":")]
    except ValueError:
        logger.warning("Invalid time format '%s' — starting from 0", time_str)
        return 0

    if not parts:
        return 0
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    logger.warning("Time '%s' has too many components — starting from 0", time_str)
    return 0
