"""Tests for TeamClassifier, BallTracker, TacticalBoard, and device resolution."""

import numpy as np
import pytest

from futsal_analytics.board import TacticalBoard
from futsal_analytics.detection import BallTracker, TeamClassifier, resolve_device

# ---------------------------------------------------------------------------
# BallTracker
# ---------------------------------------------------------------------------


class TestBallTracker:
    def test_none_input_returns_none(self):
        bt = BallTracker()
        assert bt.update(None) is None

    def test_single_position_returned(self):
        bt = BallTracker()
        pos = np.array([100.0, 200.0])
        result = bt.update(pos)
        np.testing.assert_allclose(result, pos)

    def test_smoothing_averages_positions(self):
        # Use a large max_distance so history is not reset between steps
        bt = BallTracker(max_distance=1000)
        bt.update(np.array([0.0, 0.0]))
        bt.update(np.array([100.0, 100.0]))
        result = bt.update(np.array([200.0, 200.0]))
        np.testing.assert_allclose(result, [100.0, 100.0])

    def test_large_jump_resets_history(self):
        bt = BallTracker(max_distance=50)
        bt.update(np.array([0.0, 0.0]))
        result = bt.update(np.array([1000.0, 1000.0]))
        np.testing.assert_allclose(result, [1000.0, 1000.0])

    def test_none_clears_history(self):
        """After a missed frame, smoothing should start fresh, not avg stale positions."""
        bt = BallTracker()
        bt.update(np.array([0.0, 0.0]))
        bt.update(np.array([10.0, 10.0]))
        bt.update(None)
        result = bt.update(np.array([500.0, 500.0]))
        # History was cleared by the None — result should equal the new position alone
        np.testing.assert_allclose(result, [500.0, 500.0])

    def test_ball_seen_count_increments(self):
        bt = BallTracker()
        bt.update(np.array([1.0, 1.0]))
        bt.update(np.array([2.0, 2.0]))
        assert bt.ball_seen_count == 2

    def test_none_does_not_increment_count(self):
        bt = BallTracker()
        bt.update(None)
        assert bt.ball_seen_count == 0


# ---------------------------------------------------------------------------
# TeamClassifier
# ---------------------------------------------------------------------------


class TestTeamClassifierColorFeatures:
    def test_returns_three_element_vector(self, synthetic_frame):
        tc = TeamClassifier()
        bbox = np.array([10, 10, 60, 80], dtype=np.float32)
        features = tc.get_jersey_color_features(synthetic_frame, bbox)
        assert features.shape == (3,)
        assert features.dtype == np.float32

    def test_empty_crop_returns_zeros(self, synthetic_frame):
        tc = TeamClassifier()
        bbox = np.array([10, 10, 10, 10], dtype=np.float32)
        features = tc.get_jersey_color_features(synthetic_frame, bbox)
        np.testing.assert_array_equal(features, [0, 0, 0])

    def test_predict_team_before_training_returns_zero(self, synthetic_frame):
        tc = TeamClassifier()
        bbox = np.array([10, 10, 60, 80], dtype=np.float32)
        assert tc.predict_team(synthetic_frame, bbox) == 0

    def test_safe_crop_clamps_out_of_bounds(self, synthetic_frame):
        tc = TeamClassifier()
        crop = tc._safe_crop(synthetic_frame, -100, -100, 9999, 9999)
        h, w = synthetic_frame.shape[:2]
        assert crop.shape[0] <= h
        assert crop.shape[1] <= w


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


class TestResolveDevice:
    def test_explicit_cpu(self):
        assert resolve_device("cpu") == "cpu"

    def test_explicit_cuda(self):
        # Returns "cuda" verbatim — caller is responsible for availability
        assert resolve_device("cuda") == "cuda"

    def test_auto_returns_valid_device(self):
        result = resolve_device("auto")
        assert result in ("cpu", "cuda")


# ---------------------------------------------------------------------------
# TacticalBoard
# ---------------------------------------------------------------------------


class TestTacticalBoard:
    def test_draw_state_returns_correct_shape(self):
        board = TacticalBoard(700, 350)
        img = board.draw_state([], None)
        assert img.shape == (350, 700, 3)

    def test_draw_state_with_players(self):
        board = TacticalBoard(700, 350)
        players = [
            (1, np.array([350.0, 175.0]), 0),
            (2, np.array([100.0, 100.0]), 1),
        ]
        img = board.draw_state(players, np.array([200.0, 150.0]))
        assert img.shape == (350, 700, 3)

    def test_referee_uses_grey_color(self):
        board = TacticalBoard(700, 350)
        # Referee should be drawn (unlike before), in grey
        img_no_players = board.draw_state([], None)
        img_referee = board.draw_state([(99, np.array([350.0, 175.0]), -1)], None)
        # The referee is now drawn, so the images differ
        assert not np.array_equal(img_no_players, img_referee)

    def test_team_colors_defined(self):
        assert len(TacticalBoard.TEAM_COLORS) >= 2

    def test_metres_per_pixel_scale(self):
        board = TacticalBoard(700, 350)
        # Pitch is 40m × 20m
        assert board.m_per_px_x == pytest.approx(40 / 700)
        assert board.m_per_px_y == pytest.approx(20 / 350)
