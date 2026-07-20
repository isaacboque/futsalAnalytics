"""Unit tests for pivotiq.models.data — no GPU, no weights required."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from pivotiq.models import (
    BallState,
    BBox,
    Detection,
    Event,
    EventType,
    Frame,
    Homography,
    HomographySource,
    PlayerTrack,
)


class TestBBox:
    def test_basic(self):
        b = BBox(x1=0, y1=0, x2=10, y2=20)
        assert b.width == 10.0
        assert b.height == 20.0
        assert b.area == 200.0
        assert b.center == (5.0, 10.0)
        assert b.foot_point() == (5.0, 20.0)

    def test_square(self):
        b = BBox(x1=5, y1=5, x2=10, y2=10)
        assert b.area == 25.0

    def test_invalid_ordering(self):
        with pytest.raises(ValidationError):
            BBox(x1=10, y1=0, x2=5, y2=10)   # x2 < x1
        with pytest.raises(ValidationError):
            BBox(x1=0, y1=10, x2=10, y2=5)   # y2 < y1

    def test_degenerate_zero_area(self):
        # A point bbox is technically valid
        b = BBox(x1=5, y1=5, x2=5, y2=5)
        assert b.area == 0.0


class TestDetection:
    def test_basic(self, simple_bbox_data):
        d = Detection(bbox=BBox(**simple_bbox_data), confidence=0.9, class_id=0)
        assert d.track_id is None

    def test_with_track_id(self, simple_bbox_data):
        d = Detection(bbox=BBox(**simple_bbox_data), confidence=0.5, class_id=0, track_id=42)
        assert d.track_id == 42

    def test_confidence_bounds(self, simple_bbox_data):
        with pytest.raises(ValidationError):
            Detection(bbox=BBox(**simple_bbox_data), confidence=1.5, class_id=0)
        with pytest.raises(ValidationError):
            Detection(bbox=BBox(**simple_bbox_data), confidence=-0.1, class_id=0)


class TestFrame:
    def test_basic(self):
        f = Frame(frame_idx=0, timestamp_s=0.0, width=1920, height=1080)
        assert f.width == 1920

    def test_negative_frame_idx(self):
        with pytest.raises(ValidationError):
            Frame(frame_idx=-1, timestamp_s=0.0, width=640, height=480)

    def test_zero_dimension(self):
        with pytest.raises(ValidationError):
            Frame(frame_idx=0, timestamp_s=0.0, width=0, height=480)


class TestHomography:
    def test_from_numpy_roundtrip(self, identity_H):
        h = Homography.from_numpy(frame_idx=5, H=identity_H)
        assert h.frame_idx == 5
        np.testing.assert_array_almost_equal(h.to_numpy(), identity_H)

    def test_source_default(self, identity_H):
        h = Homography.from_numpy(0, identity_H)
        assert h.source == HomographySource.KEYPOINT

    def test_wrong_shape(self):
        with pytest.raises(ValueError):
            Homography.from_numpy(0, np.eye(4))

    def test_pixel_to_pitch_identity(self, identity_H):
        h = Homography.from_numpy(0, identity_H)
        px, py = h.pixel_to_pitch(100.0, 200.0)
        assert math.isclose(px, 100.0, abs_tol=1e-6)
        assert math.isclose(py, 200.0, abs_tol=1e-6)

    def test_pitch_to_pixel_identity(self, identity_H):
        h = Homography.from_numpy(0, identity_H)
        px, py = h.pitch_to_pixel(10.0, 5.0)
        assert math.isclose(px, 10.0, abs_tol=1e-6)
        assert math.isclose(py, 5.0, abs_tol=1e-6)

    def test_json_serialisable(self, identity_H):
        h = Homography.from_numpy(0, identity_H)
        j = h.model_dump_json()
        h2 = Homography.model_validate_json(j)
        np.testing.assert_array_almost_equal(h.to_numpy(), h2.to_numpy())

    def test_h_shape_validation(self):
        with pytest.raises(ValidationError):
            Homography(frame_idx=0, H=[[1, 0], [0, 1]])  # 2×2 not 3×3

    def test_reprojection_error_non_negative(self, identity_H):
        with pytest.raises(ValidationError):
            Homography(frame_idx=0, H=identity_H.tolist(), reprojection_error=-1.0)


class TestBallState:
    def test_not_detected(self):
        b = BallState(frame_idx=10)
        assert not b.is_detected
        assert b.pixel_xy is None
        assert b.pitch_xy is None

    def test_detected(self):
        b = BallState(
            frame_idx=10,
            is_detected=True,
            pixel_xy=(320.0, 240.0),
            pitch_xy=(20.0, 10.0),
            speed_mps=8.5,
            direction_deg=45.0,
        )
        assert b.speed_mps == pytest.approx(8.5)

    def test_negative_speed(self):
        with pytest.raises(ValidationError):
            BallState(frame_idx=0, speed_mps=-1.0)


class TestPlayerTrack:
    def test_empty(self):
        pt = PlayerTrack(track_id=1)
        assert pt.last_pitch_position() is None
        assert pt.team_id is None

    def test_last_pitch_position(self, simple_bbox_data):
        det = Detection(bbox=BBox(**simple_bbox_data), confidence=0.8, class_id=0, track_id=1)
        pt = PlayerTrack(
            track_id=1,
            frame_indices=[0, 1, 2],
            detections=[det, det, det],
            pitch_positions=[None, (5.0, 8.0), (6.0, 9.0)],
        )
        assert pt.last_pitch_position() == (6.0, 9.0)

    def test_last_pitch_position_all_none(self, simple_bbox_data):
        det = Detection(bbox=BBox(**simple_bbox_data), confidence=0.8, class_id=0, track_id=2)
        pt = PlayerTrack(
            track_id=2,
            frame_indices=[0],
            detections=[det],
            pitch_positions=[None],
        )
        assert pt.last_pitch_position() is None


class TestEvent:
    def test_goal_event(self):
        e = Event(
            event_type=EventType.GOAL,
            t_seconds=65.5,
            frame_idx=1965,
            player_track_id=7,
            team_id=0,
            pitch_xy=(38.0, 10.5),
            xg=0.72,
        )
        assert e.event_type == EventType.GOAL
        assert e.xg == pytest.approx(0.72)

    def test_xg_bounds(self):
        with pytest.raises(ValidationError):
            Event(event_type=EventType.GOAL, t_seconds=0, frame_idx=0, xg=1.5)
        with pytest.raises(ValidationError):
            Event(event_type=EventType.GOAL, t_seconds=0, frame_idx=0, xg=-0.1)

    def test_json_roundtrip(self):
        e = Event(event_type=EventType.BLOCKED, t_seconds=30.0, frame_idx=900)
        e2 = Event.model_validate_json(e.model_dump_json())
        assert e2.event_type == EventType.BLOCKED

    def test_event_type_values(self):
        assert EventType.GOAL == "goal"
        assert EventType.ON_TARGET == "on_target"
        assert EventType.BLOCKED == "blocked"
        assert EventType.OFF_TARGET == "off_target"
        assert EventType.CHANCE == "chance"
