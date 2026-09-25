"""FIFA Futsal pitch geometry constants, canonical keypoints, and helpers.

Coordinate frame
----------------
  Origin : top-left corner of the pitch.
  x-axis : along the length (0 → PITCH_LENGTH_M = 40 m).
  y-axis : along the width  (0 → PITCH_WIDTH_M  = 20 m), increasing downward.

All distances in metres.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# FIFA Futsal Law 1 — pitch dimensions
# ---------------------------------------------------------------------------
PITCH_LENGTH_M: float = 40.0
PITCH_WIDTH_M: float = 20.0

GOAL_WIDTH_M: float = 3.0
GOAL_DEPTH_M: float = 1.0   # net depth behind goal line (used in 2-D polygons)
GOAL_HEIGHT_M: float = 2.0  # not used in 2-D; kept for reference

PENALTY_SPOT_1_M: float = 6.0    # first penalty spot: 6 m from goal line centre
PENALTY_SPOT_2_M: float = 10.0   # second penalty spot (red-card 10-m rule)
KEEPER_ARC_RADIUS_M: float = 6.0  # radius of the goalkeeper area arc
CENTER_CIRCLE_RADIUS_M: float = 3.0
SUBSTITUTION_ZONE_HALF_LENGTH_M: float = 5.0  # ±5 m around halfway on one touchline

# ---------------------------------------------------------------------------
# Derived constants (avoid re-computing throughout the codebase)
# ---------------------------------------------------------------------------
HALF_LENGTH: float = PITCH_LENGTH_M / 2   # 20.0
HALF_WIDTH: float = PITCH_WIDTH_M / 2     # 10.0

GOAL_POST_Y1: float = HALF_WIDTH - GOAL_WIDTH_M / 2   # 8.5  (top post)
GOAL_POST_Y2: float = HALF_WIDTH + GOAL_WIDTH_M / 2   # 11.5 (bottom post)

# Where the 6 m keeper arc meets the goal (end) line
ARC_FOOT_Y1: float = HALF_WIDTH - KEEPER_ARC_RADIUS_M   # 4.0
ARC_FOOT_Y2: float = HALF_WIDTH + KEEPER_ARC_RADIUS_M   # 16.0

# ---------------------------------------------------------------------------
# Canonical landmark keypoints (~28 points)
# Stable ordering is critical: the index of each name is its class id for
# the keypoint YOLO-pose model (see training/keypoint_detector/).
# ---------------------------------------------------------------------------
KEYPOINTS: Dict[str, Tuple[float, float]] = {
    # Corners (4)
    "corner_tl": (0.0, 0.0),
    "corner_bl": (0.0, PITCH_WIDTH_M),
    "corner_tr": (PITCH_LENGTH_M, 0.0),
    "corner_br": (PITCH_LENGTH_M, PITCH_WIDTH_M),
    # Halfway line (2)
    "halfway_t": (HALF_LENGTH, 0.0),
    "halfway_b": (HALF_LENGTH, PITCH_WIDTH_M),
    # Centre circle (5)
    "center": (HALF_LENGTH, HALF_WIDTH),
    "center_circle_t": (HALF_LENGTH, HALF_WIDTH - CENTER_CIRCLE_RADIUS_M),
    "center_circle_b": (HALF_LENGTH, HALF_WIDTH + CENTER_CIRCLE_RADIUS_M),
    "center_circle_l": (HALF_LENGTH - CENTER_CIRCLE_RADIUS_M, HALF_WIDTH),
    "center_circle_r": (HALF_LENGTH + CENTER_CIRCLE_RADIUS_M, HALF_WIDTH),
    # Left goal (x = 0 end) (6)
    "left_post_t": (0.0, GOAL_POST_Y1),
    "left_post_b": (0.0, GOAL_POST_Y2),
    "left_penalty_6m": (PENALTY_SPOT_1_M, HALF_WIDTH),
    "left_penalty_10m": (PENALTY_SPOT_2_M, HALF_WIDTH),
    "left_arc_t": (0.0, ARC_FOOT_Y1),   # arc meets end-line, top
    "left_arc_b": (0.0, ARC_FOOT_Y2),   # arc meets end-line, bottom
    # Right goal (x = 40 end) (6)
    "right_post_t": (PITCH_LENGTH_M, GOAL_POST_Y1),
    "right_post_b": (PITCH_LENGTH_M, GOAL_POST_Y2),
    "right_penalty_6m": (PITCH_LENGTH_M - PENALTY_SPOT_1_M, HALF_WIDTH),
    "right_penalty_10m": (PITCH_LENGTH_M - PENALTY_SPOT_2_M, HALF_WIDTH),
    "right_arc_t": (PITCH_LENGTH_M, ARC_FOOT_Y1),
    "right_arc_b": (PITCH_LENGTH_M, ARC_FOOT_Y2),
    # Substitution zone (4) — bottom touchline (y = 20) + symmetric refs on top
    "sub_zone_near_b": (HALF_LENGTH - SUBSTITUTION_ZONE_HALF_LENGTH_M, PITCH_WIDTH_M),
    "sub_zone_far_b": (HALF_LENGTH + SUBSTITUTION_ZONE_HALF_LENGTH_M, PITCH_WIDTH_M),
    "sub_zone_near_t": (HALF_LENGTH - SUBSTITUTION_ZONE_HALF_LENGTH_M, 0.0),
    "sub_zone_far_t": (HALF_LENGTH + SUBSTITUTION_ZONE_HALF_LENGTH_M, 0.0),
    # Goal-line centre marks (2) — useful landmark when the goal is in view
    "left_goal_centre": (0.0, HALF_WIDTH),
    "right_goal_centre": (PITCH_LENGTH_M, HALF_WIDTH),
}

KEYPOINT_NAMES: List[str] = list(KEYPOINTS.keys())
NUM_KEYPOINTS: int = len(KEYPOINT_NAMES)

# ---------------------------------------------------------------------------
# Goal-mouth polygons (pitch coordinates, metres)
# Extend GOAL_DEPTH_M behind the goal line so event code can do a simple
# point-in-polygon test to decide if the ball crossed the line.
# ---------------------------------------------------------------------------
LEFT_GOAL_POLYGON: np.ndarray = np.array(
    [
        (0.0, GOAL_POST_Y1),
        (-GOAL_DEPTH_M, GOAL_POST_Y1),
        (-GOAL_DEPTH_M, GOAL_POST_Y2),
        (0.0, GOAL_POST_Y2),
    ],
    dtype=np.float32,
)

RIGHT_GOAL_POLYGON: np.ndarray = np.array(
    [
        (PITCH_LENGTH_M, GOAL_POST_Y1),
        (PITCH_LENGTH_M + GOAL_DEPTH_M, GOAL_POST_Y1),
        (PITCH_LENGTH_M + GOAL_DEPTH_M, GOAL_POST_Y2),
        (PITCH_LENGTH_M, GOAL_POST_Y2),
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def keeper_arc_points(side: str, n: int = 64) -> np.ndarray:
    """Return *n* points along the 6 m keeper-area arc for *side* ('left'|'right').

    The arc is the half-circle of radius KEEPER_ARC_RADIUS_M centred on the
    midpoint of the respective goal line, bulging into the pitch.
    """
    if side == "left":
        cx, cy = 0.0, HALF_WIDTH
        # Angles: right semicircle (-π/2 → +π/2) — points go into positive x
        angle_start, angle_end = -math.pi / 2, math.pi / 2
    elif side == "right":
        cx, cy = PITCH_LENGTH_M, HALF_WIDTH
        # Left semicircle (π/2 → 3π/2)
        angle_start, angle_end = math.pi / 2, 3 * math.pi / 2
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    angles = np.linspace(angle_start, angle_end, n)
    xs = cx + KEEPER_ARC_RADIUS_M * np.cos(angles)
    ys = cy + KEEPER_ARC_RADIUS_M * np.sin(angles)
    return np.column_stack([xs, ys]).astype(np.float32)


def center_circle_points(n: int = 64) -> np.ndarray:
    """Return *n* points along the centre circle."""
    angles = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    xs = HALF_LENGTH + CENTER_CIRCLE_RADIUS_M * np.cos(angles)
    ys = HALF_WIDTH + CENTER_CIRCLE_RADIUS_M * np.sin(angles)
    return np.column_stack([xs, ys]).astype(np.float32)


def point_in_polygon(pt: Tuple[float, float], polygon: np.ndarray) -> bool:
    """Ray-casting containment test: is *pt* inside *polygon*?"""
    px, py = pt
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i, 0]), float(polygon[i, 1])
        xj, yj = float(polygon[j, 0]), float(polygon[j, 1])
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


def all_keypoints_array() -> np.ndarray:
    """Return all keypoints as a (NUM_KEYPOINTS, 2) float32 array in name order."""
    return np.array([KEYPOINTS[k] for k in KEYPOINT_NAMES], dtype=np.float32)
