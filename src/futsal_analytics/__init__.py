"""
Futsal Analytics
================
Real-time futsal match analysis from YouTube streams.

Pipeline: YouTube stream → YOLO detection → 6-point field calibration
→ ByteTrack tracking → team classification → perspective mapping
→ tactical board + per-player KPIs.
"""

__version__ = "0.2.0"

from futsal_analytics.board import TacticalBoard
from futsal_analytics.calibration import FieldCalibrator, load_calibration, save_calibration
from futsal_analytics.config import Config
from futsal_analytics.detection import BallTracker, TeamClassifier
from futsal_analytics.field import FieldValidator, SimpleFieldMapper
from futsal_analytics.kpis import KPITracker, PositionLogger

__all__ = [
    "Config",
    "FieldCalibrator",
    "FieldValidator",
    "SimpleFieldMapper",
    "TeamClassifier",
    "BallTracker",
    "TacticalBoard",
    "KPITracker",
    "PositionLogger",
    "load_calibration",
    "save_calibration",
]
