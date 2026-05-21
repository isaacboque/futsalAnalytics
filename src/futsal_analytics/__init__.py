"""
Futsal Analytics
================
Real-time futsal match analysis from YouTube streams.

Pipeline: YouTube stream → YOLO detection → field calibration
→ team classification → perspective mapping → tactical board.
"""

__version__ = "0.1.0"

from futsal_analytics.config import Config
from futsal_analytics.calibration import FieldCalibrator
from futsal_analytics.field import FieldValidator, SimpleFieldMapper
from futsal_analytics.detection import TeamClassifier, BallTracker
from futsal_analytics.board import TacticalBoard

__all__ = [
    "Config",
    "FieldCalibrator",
    "FieldValidator",
    "SimpleFieldMapper",
    "TeamClassifier",
    "BallTracker",
    "TacticalBoard",
]
