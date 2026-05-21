"""Tests for FieldCalibrator + load/save calibration (no GUI windows opened)."""

import numpy as np
import pytest

from futsal_analytics.calibration import (
    FieldCalibrator,
    _POINT_LABELS,
    load_calibration,
    save_calibration,
)


class TestFieldCalibratorInit:
    def test_creates_six_points(self, synthetic_frame):
        cal = FieldCalibrator(synthetic_frame)
        assert cal.points.shape == (6, 2)

    def test_point_labels_count(self):
        assert len(_POINT_LABELS) == 6

    def test_frame_is_copied(self, synthetic_frame):
        cal = FieldCalibrator(synthetic_frame)
        # Mutating the original frame should not affect the calibrator
        synthetic_frame[:] = 0
        assert cal.frame.sum() > 0

    def test_canvas_larger_than_frame(self, synthetic_frame):
        cal = FieldCalibrator(synthetic_frame)
        assert cal.canvas_w > cal.w
        assert cal.canvas_h > cal.h

    def test_initial_points_within_frame(self, synthetic_frame):
        cal = FieldCalibrator(synthetic_frame)
        assert np.all(cal.points[:, 0] >= 0)
        assert np.all(cal.points[:, 0] <= cal.w)
        assert np.all(cal.points[:, 1] >= 0)
        assert np.all(cal.points[:, 1] <= cal.h)


class TestGetPointIndex:
    def test_hit_returns_index(self, synthetic_frame):
        cal = FieldCalibrator(synthetic_frame)
        x, y = int(cal.points[0][0]), int(cal.points[0][1])
        assert cal.get_point_index(x, y) == 0

    def test_miss_returns_none(self, synthetic_frame):
        cal = FieldCalibrator(synthetic_frame)
        assert cal.get_point_index(-500, -500) is None

    def test_nearest_wins(self, synthetic_frame):
        cal = FieldCalibrator(synthetic_frame)
        # Query exactly on point 3
        x, y = int(cal.points[3][0]), int(cal.points[3][1])
        assert cal.get_point_index(x, y) == 3


class TestUpdatePoint:
    def test_moves_point(self, synthetic_frame):
        cal = FieldCalibrator(synthetic_frame)
        cal.update_point(2, 999, 888)
        assert cal.points[2][0] == pytest.approx(999.0)
        assert cal.points[2][1] == pytest.approx(888.0)

    def test_other_points_unchanged(self, synthetic_frame):
        cal = FieldCalibrator(synthetic_frame)
        original = cal.points.copy()
        cal.update_point(0, 1, 1)
        np.testing.assert_array_equal(cal.points[1:], original[1:])


class TestDrawFrame:
    def test_returns_correct_shape(self, synthetic_frame):
        cal = FieldCalibrator(synthetic_frame)
        canvas = cal.draw_frame()
        assert canvas.shape == (cal.canvas_h, cal.canvas_w, 3)
        assert canvas.dtype == np.uint8


class TestLoadSaveCalibration:
    def test_round_trip(self, tmp_path, default_polygon):
        path = tmp_path / "cal.npy"
        save_calibration(default_polygon, path)
        loaded = load_calibration(path)
        np.testing.assert_array_almost_equal(loaded, default_polygon)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_calibration(tmp_path / "missing.npy")

    def test_too_few_points_raises(self, tmp_path):
        path = tmp_path / "few.npy"
        np.save(path, np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32))
        with pytest.raises(ValueError):
            load_calibration(path)

    def test_wrong_shape_raises(self, tmp_path):
        path = tmp_path / "wrong.npy"
        np.save(path, np.array([0, 1, 2, 3, 4, 5], dtype=np.float32))
        with pytest.raises(ValueError):
            load_calibration(path)
