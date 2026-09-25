"""Pydantic v2 data contracts for PivotIQ.

All cross-module data flows through these models so the pipeline is
type-safe and JSON-serialisable end-to-end.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class BBox(BaseModel):
    """Axis-aligned bounding box in pixel space."""

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _check_ordering(self) -> "BBox":
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("x2/y2 must be >= x1/y1")
        return self

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre of the box — used as the player's ground position."""
        return ((self.x1 + self.x2) / 2.0, self.y2)


# ---------------------------------------------------------------------------
# Per-frame primitives
# ---------------------------------------------------------------------------


class Detection(BaseModel):
    """Single object detection in one frame."""

    bbox: BBox
    confidence: float = Field(ge=0.0, le=1.0)
    class_id: int
    track_id: Optional[int] = None


class Frame(BaseModel):
    """Metadata for a single decoded video frame."""

    frame_idx: int = Field(ge=0)
    timestamp_s: float = Field(ge=0.0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------


class PlayerTrack(BaseModel):
    """Accumulated track for one player across multiple frames."""

    track_id: int
    team_id: Optional[int] = None
    # Parallel lists — one entry per frame the player was detected.
    frame_indices: list[int] = Field(default_factory=list)
    detections: list[Detection] = Field(default_factory=list)
    # Pitch-space position at each detection frame; None when registration failed.
    pitch_positions: list[Optional[tuple[float, float]]] = Field(default_factory=list)

    def last_pitch_position(self) -> Optional[tuple[float, float]]:
        for pos in reversed(self.pitch_positions):
            if pos is not None:
                return pos
        return None


# ---------------------------------------------------------------------------
# Ball
# ---------------------------------------------------------------------------


class BallState(BaseModel):
    """Ball state at a single frame (may be interpolated by Kalman filter)."""

    frame_idx: int = Field(ge=0)
    is_detected: bool = False
    pixel_xy: Optional[tuple[float, float]] = None
    pitch_xy: Optional[tuple[float, float]] = None
    speed_mps: Optional[float] = Field(default=None, ge=0.0)
    direction_deg: Optional[float] = None  # 0=right, 90=down (image coords)


# ---------------------------------------------------------------------------
# Registration / homography
# ---------------------------------------------------------------------------


class HomographySource(str, Enum):
    KEYPOINT = "keypoint"
    OPTICAL_FLOW = "optical_flow"
    PROPAGATED = "propagated"
    MOCK = "mock"


class Homography(BaseModel):
    """3×3 homography mapping pixel space → pitch space for one frame."""

    frame_idx: int = Field(ge=0)
    # Stored as nested list for JSON serialisability; use to_numpy() for math.
    H: list[list[float]] = Field(description="Row-major 3×3 matrix")
    reprojection_error: Optional[float] = Field(default=None, ge=0.0)
    source: HomographySource = HomographySource.KEYPOINT

    @field_validator("H")
    @classmethod
    def _check_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("H must be a 3×3 matrix")
        return v

    @classmethod
    def from_numpy(
        cls,
        frame_idx: int,
        H: np.ndarray,
        reprojection_error: Optional[float] = None,
        source: HomographySource = HomographySource.KEYPOINT,
    ) -> "Homography":
        if H.shape != (3, 3):
            raise ValueError(f"Expected (3,3) array, got {H.shape}")
        return cls(
            frame_idx=frame_idx,
            H=H.tolist(),
            reprojection_error=reprojection_error,
            source=source,
        )

    def to_numpy(self) -> np.ndarray:
        return np.array(self.H, dtype=np.float64)

    def pixel_to_pitch(self, px: float, py: float) -> tuple[float, float]:
        """Map one pixel point to pitch coordinates."""
        H = self.to_numpy()
        v = H @ np.array([px, py, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    def pitch_to_pixel(self, mx: float, my: float) -> tuple[float, float]:
        """Map one pitch point to pixel coordinates (uses inverse H)."""
        H_inv = np.linalg.inv(self.to_numpy())
        v = H_inv @ np.array([mx, my, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    GOAL = "goal"
    ON_TARGET = "on_target"
    BLOCKED = "blocked"
    OFF_TARGET = "off_target"
    CHANCE = "chance"


class Event(BaseModel):
    """A detected match event (shot, goal, big chance, etc.)."""

    event_type: EventType
    t_seconds: float = Field(ge=0.0)
    frame_idx: int = Field(ge=0)
    player_track_id: Optional[int] = None
    team_id: Optional[int] = None
    pitch_xy: Optional[tuple[float, float]] = None  # ball/shot origin in pitch coords
    xg: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    clip_path: Optional[str] = None  # path to 30 s buildup clip
