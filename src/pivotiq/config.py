"""PivotIQ runtime configuration.

All thresholds and tunable parameters live here so they can be overridden
from a TOML file without touching source code.

Usage::

    from pivotiq.config import PivotIQConfig
    cfg = PivotIQConfig.from_toml(Path("pivotiq.toml"))
    # or:
    cfg = PivotIQConfig.default()
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]  # backport for ≤3.10


@dataclass
class DetectionConfig:
    """YOLO detection thresholds and model paths."""

    player_model: str = "yolo11n.pt"
    ball_model: Optional[str] = None   # None → falls back to player_model + warning
    player_confidence: float = 0.40
    ball_confidence: float = 0.30
    device: str = "auto"               # "auto" | "cpu" | "cuda" | "mps"
    player_class_id: int = 0           # COCO person class


@dataclass
class TrackingConfig:
    """ByteTrack player tracking parameters."""

    track_thresh: float = 0.50
    track_buffer: int = 30             # frames to keep lost track alive
    match_thresh: float = 0.80
    min_box_area: float = 10.0         # px² — ignore tiny detections


@dataclass
class RegistrationConfig:
    """Dynamic field registration parameters."""

    min_visible_keypoints: int = 4     # RANSAC needs ≥4 point correspondences
    ransac_reproj_threshold: float = 4.0   # pixels
    temporal_smoothing_alpha: float = 0.70  # exponential smoothing for H propagation
    optical_flow_max_corners: int = 200
    optical_flow_quality: float = 0.01


@dataclass
class EventConfig:
    """Shot / goal / chance detection thresholds."""

    shot_speed_threshold_mps: float = 5.0   # min ball speed to flag as a shot
    chance_xg_threshold: float = 0.20       # xG above this → big chance
    buildup_seconds: float = 30.0           # pre-event clip length
    goal_rebound_frames: int = 10           # frames to look for rebound after goal line


@dataclass
class PivotIQConfig:
    """Top-level config object; one instance per analysis run."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    events: EventConfig = field(default_factory=EventConfig)

    @classmethod
    def default(cls) -> "PivotIQConfig":
        return cls()

    @classmethod
    def from_toml(cls, path: Path) -> "PivotIQConfig":
        """Load config from a TOML file; unset keys keep their defaults."""
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        cfg = cls()
        _section_map = {
            "detection": (DetectionConfig, "detection"),
            "tracking": (TrackingConfig, "tracking"),
            "registration": (RegistrationConfig, "registration"),
            "events": (EventConfig, "events"),
        }
        for key, (dcls, attr) in _section_map.items():
            if key in raw:
                current = asdict(getattr(cfg, attr))
                current.update(raw[key])
                setattr(cfg, attr, dcls(**current))
        return cfg
