"""Tests for stream.parse_start_time and the local-file branch of open_youtube_stream."""


import numpy as np
import pytest

from futsal_analytics.config import Config
from futsal_analytics.stream import open_youtube_stream, parse_start_time


class TestParseStartTime:
    def test_zero(self):
        assert parse_start_time("0") == 0

    def test_seconds_only(self):
        assert parse_start_time("45") == 45

    def test_mm_ss(self):
        assert parse_start_time("1:30") == 90
        assert parse_start_time("17:00") == 17 * 60

    def test_hh_mm_ss(self):
        # New: HH:MM:SS support (was previously silently truncated)
        assert parse_start_time("1:30:00") == 3600 + 1800
        assert parse_start_time("0:01:30") == 90
        assert parse_start_time("2:00:00") == 7200

    def test_whitespace_stripped(self):
        assert parse_start_time("  10:00  ") == 600

    def test_garbage_returns_zero(self):
        assert parse_start_time("not-a-time") == 0
        assert parse_start_time("abc:def") == 0

    def test_too_many_components(self):
        assert parse_start_time("1:2:3:4") == 0

    def test_none_input(self):
        assert parse_start_time(None) == 0


class TestOpenYouTubeStreamLocalFile:
    """The local-file fast path should not require yt-dlp to be installed."""

    def test_returns_none_for_missing_path(self):
        cap = open_youtube_stream("definitely-not-a-real-path.mp4", Config())
        # Either rejected as a file (None) or attempted via yt-dlp (also None)
        assert cap is None

    def test_opens_local_mp4(self, tmp_path):
        # Synthesize a tiny MP4 with OpenCV so we don't need a fixture asset.
        import cv2

        video_path = tmp_path / "tiny.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (32, 32),
        )
        if not writer.isOpened():
            pytest.skip("Local mp4v codec not available on this platform")
        for _ in range(5):
            writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
        writer.release()

        cap = open_youtube_stream(str(video_path), Config())
        assert cap is not None
        assert cap.isOpened()
        ret, frame = cap.read()
        cap.release()
        assert ret is True
        assert frame is not None
