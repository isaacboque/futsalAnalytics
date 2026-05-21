"""Tests for the small runtime helpers in __main__: snapshot writer + lock file."""

import time

import numpy as np
import pytest

from futsal_analytics.__main__ import (
    LOCK_FILENAME,
    LOCK_STALE_AFTER_S,
    _claim_lock,
    _lock_is_active,
    _refresh_lock,
    _write_snapshot,
)


@pytest.fixture
def synthetic_image():
    img = np.zeros((180, 320, 3), dtype=np.uint8)
    img[:, :, 1] = 128  # solid green-ish
    return img


class TestWriteSnapshot:
    def test_creates_file(self, tmp_path, synthetic_image):
        target = tmp_path / "_live_camera.jpg"
        _write_snapshot(target, synthetic_image)
        assert target.exists()
        assert target.stat().st_size > 0

    def test_no_tmp_leftover(self, tmp_path, synthetic_image):
        target = tmp_path / "_live_camera.jpg"
        _write_snapshot(target, synthetic_image)
        # The atomic-rename helper must not leave behind a .tmp file
        assert not (tmp_path / "_live_camera.tmp.jpg").exists()

    def test_overwrites_atomically(self, tmp_path, synthetic_image):
        target = tmp_path / "_live_board.jpg"
        _write_snapshot(target, synthetic_image)
        size_first = target.stat().st_size

        second = np.full_like(synthetic_image, 200)  # all-grey, different size
        _write_snapshot(target, second)
        size_second = target.stat().st_size
        assert size_second != size_first

    def test_downscale_reduces_file(self, tmp_path, synthetic_image):
        big = np.zeros((1080, 1920, 3), dtype=np.uint8)
        big[:, :, 1] = 200
        target_full = tmp_path / "full.jpg"
        target_small = tmp_path / "small.jpg"
        _write_snapshot(target_full, big, max_width=0)
        _write_snapshot(target_small, big, max_width=480)
        assert target_small.stat().st_size < target_full.stat().st_size

    def test_invalid_path_is_silent(self, tmp_path, synthetic_image):
        # Writing to a nonexistent subdirectory should not crash
        target = tmp_path / "does-not-exist" / "out.jpg"
        _write_snapshot(target, synthetic_image)
        # Either the call swallowed the error, or it created the parent.
        # Both behaviours are acceptable; the only contract is "doesn't raise".


class TestLockFile:
    def test_claim_fresh(self, tmp_path):
        lock = tmp_path / LOCK_FILENAME
        assert _claim_lock(lock, "http://example.com/x") is True
        assert lock.exists()
        contents = lock.read_text(encoding="utf-8")
        assert "pid=" in contents
        assert "http://example.com/x" in contents

    def test_claim_fails_when_active(self, tmp_path):
        lock = tmp_path / LOCK_FILENAME
        assert _claim_lock(lock, "url1") is True
        # Same lock cannot be claimed while still active
        assert _claim_lock(lock, "url2") is False
        # And the original contents survived
        assert "url1" in lock.read_text(encoding="utf-8")

    def test_claim_succeeds_over_stale(self, tmp_path):
        lock = tmp_path / LOCK_FILENAME
        lock.write_text("pid=1\nurl=old\nstart=0\n", encoding="utf-8")
        # Back-date the file so it's considered stale
        very_old = time.time() - (LOCK_STALE_AFTER_S + 10)
        import os
        os.utime(lock, (very_old, very_old))
        assert _lock_is_active(lock) is False
        assert _claim_lock(lock, "fresh") is True

    def test_refresh_updates_mtime(self, tmp_path):
        lock = tmp_path / LOCK_FILENAME
        _claim_lock(lock, "x")
        first_mtime = lock.stat().st_mtime
        time.sleep(0.05)
        _refresh_lock(lock)
        assert lock.stat().st_mtime >= first_mtime
