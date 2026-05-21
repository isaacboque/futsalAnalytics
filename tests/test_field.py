"""Tests for FieldValidator and SimpleFieldMapper."""

import numpy as np
import pytest

from futsal_analytics.field import FieldValidator, SimpleFieldMapper


# ---------------------------------------------------------------------------
# FieldValidator
# ---------------------------------------------------------------------------


class TestFieldValidatorInit:
    def test_accepts_six_points(self, default_polygon):
        fv = FieldValidator(default_polygon)
        assert fv.field_polygon.shape == (6, 2)

    def test_raises_with_fewer_than_four_points(self):
        with pytest.raises(ValueError):
            FieldValidator(np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32))

    def test_bounding_box_computed(self, default_polygon):
        fv = FieldValidator(default_polygon)
        assert fv.x_min < fv.x_max
        assert fv.y_min < fv.y_max


class TestIsWithinField:
    def test_centre_of_polygon_is_inside(self, default_polygon):
        fv = FieldValidator(default_polygon)
        centre = np.mean(default_polygon, axis=0)
        assert fv.is_within_field(centre)

    def test_far_outside_is_rejected(self, default_polygon):
        fv = FieldValidator(default_polygon)
        assert not fv.is_within_field(np.array([-1000.0, -1000.0]))

    def test_edge_point_is_accepted(self, default_polygon):
        fv = FieldValidator(default_polygon)
        # TL corner is on the boundary
        assert fv.is_within_field(default_polygon[0])


class TestFilterDetections:
    """Uses a minimal mock for supervision.Detections (avoids the dependency)."""

    class _MockDetections:
        def __init__(self, bboxes: np.ndarray):
            self.xyxy = bboxes

    def test_inside_detections_kept(self, default_polygon):
        fv = FieldValidator(default_polygon)
        centre = np.mean(default_polygon, axis=0)
        bboxes = np.array(
            [[centre[0] - 5, centre[1] - 10, centre[0] + 5, centre[1]]],
            dtype=np.float32,
        )
        dets = self._MockDetections(bboxes)
        feet = np.array([centre])
        mask = fv.filter_detections(dets, feet)
        assert mask[0] is np.bool_(True)

    def test_outside_detections_rejected(self, default_polygon):
        fv = FieldValidator(default_polygon)
        bboxes = np.array([[-500.0, -500.0, -490.0, -490.0]], dtype=np.float32)
        dets = self._MockDetections(bboxes)
        feet = np.array([[-500.0, -500.0]])
        mask = fv.filter_detections(dets, feet)
        assert mask[0] is np.bool_(False)


# ---------------------------------------------------------------------------
# SimpleFieldMapper
# ---------------------------------------------------------------------------


class TestSimpleFieldMapper:
    def test_corner_maps_to_board_corner(self, default_polygon):
        board_w, board_h = 700, 350
        mapper = SimpleFieldMapper(default_polygon, board_w, board_h)
        # TL (index 0) → board (0, 0)
        result = mapper.transform(default_polygon[0])
        assert result[0] == pytest.approx(0.0, abs=2.0)
        assert result[1] == pytest.approx(0.0, abs=2.0)

    def test_tr_corner_maps_to_board_top_right(self, default_polygon):
        board_w, board_h = 700, 350
        mapper = SimpleFieldMapper(default_polygon, board_w, board_h)
        # TR (index 2) → board (board_w, 0)
        result = mapper.transform(default_polygon[2])
        assert result[0] == pytest.approx(board_w, abs=2.0)
        assert result[1] == pytest.approx(0.0, abs=2.0)

    def test_invalid_point_returns_zeros(self, default_polygon):
        mapper = SimpleFieldMapper(default_polygon, 700, 350)
        result = mapper.transform(np.array([-1.0, -1.0]))
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_accepts_fewer_than_six_points(self):
        four_pts = np.array(
            [[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32
        )
        mapper = SimpleFieldMapper(four_pts, 700, 350)
        assert mapper.M is not None
