"""Unit tests for pivotiq.geometry — no GPU, no weights required."""

import math

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import

import matplotlib.pyplot as plt

from pivotiq.geometry import (
    ARC_FOOT_Y1,
    ARC_FOOT_Y2,
    CENTER_CIRCLE_RADIUS_M,
    GOAL_POST_Y1,
    GOAL_POST_Y2,
    GOAL_WIDTH_M,
    HALF_LENGTH,
    HALF_WIDTH,
    KEEPER_ARC_RADIUS_M,
    KEYPOINT_NAMES,
    KEYPOINTS,
    LEFT_GOAL_POLYGON,
    NUM_KEYPOINTS,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    RIGHT_GOAL_POLYGON,
    center_circle_points,
    keeper_arc_points,
    point_in_polygon,
)
from pivotiq.geometry.draw import draw_pitch, draw_shot_map
from pivotiq.geometry.pitch import all_keypoints_array


class TestPitchConstants:
    def test_dimensions(self):
        assert PITCH_LENGTH_M == pytest.approx(40.0)
        assert PITCH_WIDTH_M == pytest.approx(20.0)

    def test_goal_width(self):
        assert GOAL_WIDTH_M == pytest.approx(3.0)
        assert GOAL_POST_Y1 == pytest.approx(HALF_WIDTH - 1.5)
        assert GOAL_POST_Y2 == pytest.approx(HALF_WIDTH + 1.5)

    def test_half_constants(self):
        assert HALF_LENGTH == pytest.approx(20.0)
        assert HALF_WIDTH == pytest.approx(10.0)

    def test_arc_feet(self):
        assert ARC_FOOT_Y1 == pytest.approx(HALF_WIDTH - KEEPER_ARC_RADIUS_M)
        assert ARC_FOOT_Y2 == pytest.approx(HALF_WIDTH + KEEPER_ARC_RADIUS_M)


class TestKeypoints:
    def test_count_in_spec_range(self):
        """Spec requires ~25–30 canonical landmarks."""
        assert 25 <= NUM_KEYPOINTS <= 32, f"NUM_KEYPOINTS={NUM_KEYPOINTS} out of range"

    def test_names_match_dict(self):
        assert KEYPOINT_NAMES == list(KEYPOINTS.keys())

    def test_all_within_pitch_bounds_or_on_boundary(self):
        tol = 0.01  # metre
        for name, (x, y) in KEYPOINTS.items():
            assert -tol <= x <= PITCH_LENGTH_M + tol, f"{name} x={x} out of range"
            assert -tol <= y <= PITCH_WIDTH_M + tol, f"{name} y={y} out of range"

    def test_corner_positions(self):
        assert KEYPOINTS["corner_tl"] == pytest.approx((0.0, 0.0))
        assert KEYPOINTS["corner_br"] == pytest.approx((PITCH_LENGTH_M, PITCH_WIDTH_M))
        assert KEYPOINTS["corner_tr"] == pytest.approx((PITCH_LENGTH_M, 0.0))
        assert KEYPOINTS["corner_bl"] == pytest.approx((0.0, PITCH_WIDTH_M))

    def test_center_position(self):
        assert KEYPOINTS["center"] == pytest.approx((HALF_LENGTH, HALF_WIDTH))

    def test_penalty_spots_on_midline(self):
        assert KEYPOINTS["left_penalty_6m"][1] == pytest.approx(HALF_WIDTH)
        assert KEYPOINTS["right_penalty_6m"][1] == pytest.approx(HALF_WIDTH)

    def test_penalty_spot_distances(self):
        lx, _ = KEYPOINTS["left_penalty_6m"]
        rx, _ = KEYPOINTS["right_penalty_6m"]
        assert lx == pytest.approx(6.0)
        assert rx == pytest.approx(34.0)

    def test_unique_positions(self):
        positions = list(KEYPOINTS.values())
        unique = set(positions)
        assert len(positions) == len(unique), "Duplicate keypoint coordinates found"

    def test_all_keypoints_array_shape(self):
        arr = all_keypoints_array()
        assert arr.shape == (NUM_KEYPOINTS, 2)
        assert arr.dtype == np.float32


class TestGoalPolygons:
    def test_left_goal_contains_inside_point(self):
        # (-0.5, HALF_WIDTH) is inside the left goal net
        assert point_in_polygon((-0.5, HALF_WIDTH), LEFT_GOAL_POLYGON)

    def test_right_goal_contains_inside_point(self):
        # (PITCH_LENGTH_M + 0.5, HALF_WIDTH) is inside the right goal net
        assert point_in_polygon(
            (PITCH_LENGTH_M + 0.5, HALF_WIDTH), RIGHT_GOAL_POLYGON
        )

    def test_center_pitch_not_in_either_goal(self):
        assert not point_in_polygon((HALF_LENGTH, HALF_WIDTH), LEFT_GOAL_POLYGON)
        assert not point_in_polygon((HALF_LENGTH, HALF_WIDTH), RIGHT_GOAL_POLYGON)

    def test_post_position_on_goal_boundary(self):
        # A point exactly on the goal-line (x=0) at post height
        # is on the boundary — not strictly inside, so we test near-inside
        assert point_in_polygon((-0.01, GOAL_POST_Y1 + 0.1), LEFT_GOAL_POLYGON)

    def test_off_target_not_in_goal(self):
        # A point well outside the post is not in the goal
        assert not point_in_polygon((-0.5, 0.0), LEFT_GOAL_POLYGON)


class TestArcHelpers:
    def test_keeper_arc_left_shape(self):
        pts = keeper_arc_points("left", n=32)
        assert pts.shape == (32, 2)
        assert pts.dtype == np.float32

    def test_keeper_arc_right_shape(self):
        pts = keeper_arc_points("right", n=64)
        assert pts.shape == (64, 2)

    def test_keeper_arc_left_endpoints(self):
        pts = keeper_arc_points("left", n=64)
        # Start: (0, ARC_FOOT_Y1), End: (0, ARC_FOOT_Y2)
        assert pts[0, 0] == pytest.approx(0.0, abs=0.01)
        assert pts[0, 1] == pytest.approx(ARC_FOOT_Y1, abs=0.01)
        assert pts[-1, 0] == pytest.approx(0.0, abs=0.01)
        assert pts[-1, 1] == pytest.approx(ARC_FOOT_Y2, abs=0.01)

    def test_keeper_arc_left_farthest_point(self):
        # With n=64 discrete points the farthest-x sample lands near angle=0
        # (KEEPER_ARC_RADIUS_M, HALF_WIDTH) but not exactly — use 0.25 m tol.
        pts = keeper_arc_points("left")
        idx_max_x = int(np.argmax(pts[:, 0]))
        assert pts[idx_max_x, 0] == pytest.approx(KEEPER_ARC_RADIUS_M, abs=0.25)
        assert pts[idx_max_x, 1] == pytest.approx(HALF_WIDTH, abs=0.25)

    def test_keeper_arc_invalid_side(self):
        with pytest.raises(ValueError):
            keeper_arc_points("top")

    def test_center_circle_shape(self):
        pts = center_circle_points(n=32)
        assert pts.shape == (32, 2)

    def test_center_circle_radius(self):
        pts = center_circle_points()
        distances = np.sqrt(
            (pts[:, 0] - HALF_LENGTH) ** 2 + (pts[:, 1] - HALF_WIDTH) ** 2
        )
        np.testing.assert_allclose(distances, CENTER_CIRCLE_RADIUS_M, atol=0.01)


class TestPointInPolygon:
    def test_unit_square_inside(self):
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        assert point_in_polygon((0.5, 0.5), square)

    def test_unit_square_outside(self):
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        assert not point_in_polygon((2.0, 0.5), square)

    def test_triangle(self):
        tri = np.array([[0, 0], [4, 0], [2, 3]], dtype=np.float32)
        assert point_in_polygon((2.0, 1.0), tri)
        assert not point_in_polygon((0.0, 2.0), tri)


class TestDrawPitch:
    def test_returns_fig_ax(self):
        fig, ax = draw_pitch()
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_custom_figsize(self):
        fig, ax = draw_pitch(figsize=(8.0, 4.0))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(8.0)
        assert h == pytest.approx(4.0)
        plt.close(fig)

    def test_into_existing_axes(self):
        fig, ax = plt.subplots()
        fig2, ax2 = draw_pitch(ax=ax)
        assert fig2 is fig
        assert ax2 is ax
        plt.close(fig)

    def test_shot_map_no_crash(self):
        shots = [(5.0, 10.0), (38.0, 9.0), (30.0, 12.0)]
        outcomes = ["goal", "on_target", "blocked"]
        fig, ax = draw_shot_map(shots, outcomes)
        assert fig is not None
        plt.close(fig)

    def test_shot_map_empty(self):
        fig, ax = draw_shot_map([], [])
        assert fig is not None
        plt.close(fig)
