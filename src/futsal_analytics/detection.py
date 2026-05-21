"""
Player and ball detection, team classification, and per-frame processing.

Heavy imports (ultralytics, supervision, sklearn) are loaded lazily so that
calibration-only workflows start quickly without loading the full ML stack.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from futsal_analytics.board import TacticalBoard
from futsal_analytics.config import Config
from futsal_analytics.field import FieldValidator, SimpleFieldMapper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Team Classifier
# ---------------------------------------------------------------------------


class TeamClassifier:
    """Assigns players to teams via K-Means clustering on HSV jersey colour."""

    def __init__(self, n_clusters: int = 3, n_init: int = 20) -> None:
        from sklearn.cluster import KMeans

        self.kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
        self.trained: bool = False
        self.ref_label: Optional[int] = None

    @staticmethod
    def _safe_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=frame.dtype)
        return frame[y1:y2, x1:x2]

    def get_jersey_color_features(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract a 3-element HSV feature vector from the upper portion of a bounding box."""
        x1, y1, x2, y2 = map(int, bbox)
        jersey_h = int((y2 - y1) * 0.4)
        crop = self._safe_crop(frame, x1, y1, x2, y1 + jersey_h)

        if crop.size == 0:
            return np.array([0, 0, 0], dtype=np.float32)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat_mask = hsv[:, :, 1] > 30

        if sat_mask.sum() < 5:
            return np.mean(hsv, axis=(0, 1)).astype(np.float32)

        features = np.mean(hsv[sat_mask], axis=0).astype(np.float32)
        features[0] = np.clip(features[0], 0, 180)
        return features

    def train_teams(self, frame: np.ndarray, detections: Any, config: Config) -> bool:
        """
        Fit K-Means on jersey colours of the currently visible players.

        Returns True on success.
        """
        if len(detections) < config.min_players_for_kmeans:
            logger.debug(
                "[TRAIN] Not enough players (%d/%d)",
                len(detections),
                config.min_players_for_kmeans,
            )
            return False

        try:
            features = np.array(
                [self.get_jersey_color_features(frame, b) for b in detections.xyxy],
                dtype=np.float32,
            )

            hue_var = float(np.var(features[:, 0]))
            if hue_var < 50:
                logger.warning("[TRAIN] Insufficient colour variance (hue var=%.1f)", hue_var)
                return False

            self.kmeans.fit(features)
            self.trained = True
            counts = np.bincount(self.kmeans.labels_)
            self.ref_label = int(np.argmin(counts))

            logger.info("[TRAIN] K-Means trained. Referee cluster=%d, distribution=%s", self.ref_label, list(counts))
            return True

        except Exception as exc:
            logger.error("[TRAIN] Error: %s", exc)
            return False

    def predict_team(self, frame: np.ndarray, bbox: np.ndarray) -> int:
        """
        Return 0 or 1 for team assignment, or -1 for referee.

        Falls back to team 0 if the classifier has not been trained yet.
        """
        if not self.trained:
            return 0
        try:
            features = self.get_jersey_color_features(frame, bbox)
            label = int(self.kmeans.predict(features.reshape(1, -1))[0])
            if label == self.ref_label:
                return -1
            team_labels = [l for l in range(self.kmeans.n_clusters) if l != self.ref_label]
            return team_labels.index(label) if label in team_labels else 0
        except Exception as exc:
            logger.warning("Team prediction error: %s", exc)
            return 0


# ---------------------------------------------------------------------------
# Ball Tracker
# ---------------------------------------------------------------------------


class BallTracker:
    """Simple frame-to-frame ball tracker with temporal position smoothing."""

    def __init__(self, max_distance: float = 100) -> None:
        """
        Args:
            max_distance: Maximum pixel displacement between consecutive frames
                          before the position history is reset.
        """
        self.last_pos: Optional[np.ndarray] = None
        self.max_distance = max_distance
        self.ball_seen_count: int = 0
        self._history: List[np.ndarray] = []

    def update(self, current_pos: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Update with the ball's detected position this frame.

        Args:
            current_pos: Detected ball centre, or None if not visible.

        Returns:
            Smoothed position (mean of recent positions), or None.
        """
        if current_pos is None:
            self.last_pos = None
            return None

        if self.last_pos is not None:
            dist = float(np.linalg.norm(current_pos - self.last_pos))
            if dist > self.max_distance:
                logger.debug("Ball discontinuity detected (dist=%.1f) — resetting history", dist)
                self._history = []

        self.last_pos = current_pos.copy()
        self.ball_seen_count += 1
        self._history.append(current_pos.copy())
        if len(self._history) > 5:
            self._history = self._history[-5:]

        return np.mean(self._history, axis=0)


# ---------------------------------------------------------------------------
# YOLO setup
# ---------------------------------------------------------------------------


def setup_detectors(config: Config) -> Optional[Any]:
    """
    Load the YOLO model specified in *config*.

    Heavy imports are deferred to this call so the calibration UI starts
    without loading the full ML stack.

    Returns:
        A ``ultralytics.YOLO`` instance, or ``None`` on failure.
    """
    logger.info("[SETUP] Loading ML dependencies (ultralytics)...")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        logger.error("[SETUP] Failed to import ultralytics: %s", exc)
        return None

    try:
        model = YOLO(config.model_name)
        logger.info("[SETUP] YOLO model '%s' loaded", config.model_name)
        return model
    except Exception as exc:
        logger.error("[SETUP] Failed to load YOLO model: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Per-frame processing pipeline
# ---------------------------------------------------------------------------


def process_frame(
    frame: np.ndarray,
    model: Any,
    classifier: TeamClassifier,
    mapper: SimpleFieldMapper,
    field_validator: FieldValidator,
    board_drawer: TacticalBoard,
    ball_tracker: BallTracker,
    config: Config,
    fps: float,  # noqa: ARG001  (reserved for future KPI use)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the full per-frame detection pipeline without persistent player IDs.

    Steps:
        1. YOLO detection.
        2. Size filter (discard tiny bboxes).
        3. Field polygon filter (discard detections outside the pitch).
        4. Team classification (lazy K-Means training on first sufficient batch).
        5. Perspective mapping to board coordinates.
        6. Ball tracking with temporal smoothing.
        7. Annotation of camera frame and tactical board rendering.

    Args:
        frame: Raw BGR video frame.
        model: Loaded YOLO model.
        classifier: Team colour classifier.
        mapper: Perspective mapper (camera → board).
        field_validator: Pitch polygon validator.
        board_drawer: Tactical board renderer.
        ball_tracker: Ball position smoother.
        config: System configuration.
        fps: Video frame rate (reserved for future KPI calculations).

    Returns:
        Tuple of (annotated_camera_frame, tactical_board_image).
    """
    import supervision as sv

    results = model.predict(
        frame,
        classes=[config.player_class_id, config.ball_class_id],
        conf=config.yolo_conf_threshold,
        verbose=False,
        iou=0.5,
    )[0]

    detections = sv.Detections.from_ultralytics(results)
    player_dets = detections[detections.class_id == config.player_class_id]
    ball_dets = detections[detections.class_id == config.ball_class_id]

    logger.info("[DETECT] Players (raw)=%d  Ball=%d", len(player_dets), len(ball_dets))

    # Size filter
    if len(player_dets) > 0:
        areas = (player_dets.xyxy[:, 2] - player_dets.xyxy[:, 0]) * (
            player_dets.xyxy[:, 3] - player_dets.xyxy[:, 1]
        )
        player_dets = player_dets[areas > 300]

    # Pitch polygon filter
    if len(player_dets) > 0:
        feet = player_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        valid_mask = field_validator.filter_detections(player_dets, feet)
        logger.info("[FILTER] Removed=%d  Inside field=%d", int(np.sum(~valid_mask)), int(np.sum(valid_mask)))
        player_dets = player_dets[valid_mask]

    # Train classifier on the first frame with enough players
    if not classifier.trained and len(player_dets) >= config.min_players_for_kmeans:
        classifier.train_teams(frame, player_dets, config)

    # Map players to board
    players_frame: List[Tuple[int, np.ndarray, int]] = []
    if len(player_dets) > 0:
        feet = player_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        team_counts: Dict[str, int] = {}
        for i, foot in enumerate(feet):
            team_id = classifier.predict_team(frame, player_dets.xyxy[i])
            mapped = mapper.transform(foot)
            players_frame.append((i + 1, mapped, team_id))
            key = f"Team {team_id}" if team_id >= 0 else "Referee"
            team_counts[key] = team_counts.get(key, 0) + 1
        logger.info("[MAP] %d players on board | distribution: %s", len(player_dets), team_counts)

    # Ball
    current_ball: Optional[np.ndarray] = None
    if len(ball_dets) > 0:
        ball_center = ball_dets.get_anchors_coordinates(sv.Position.CENTER)[0]
        if field_validator.is_within_field(ball_center):
            current_ball = mapper.transform(ball_center)
    mapped_ball = ball_tracker.update(current_ball)

    # Annotate camera frame
    annotated = frame.copy()
    for i, bbox in enumerate(player_dets.xyxy):
        team_id = classifier.predict_team(frame, bbox)
        color = (128, 128, 128) if team_id < 0 else TacticalBoard.TEAM_COLORS[team_id % 2]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

    if len(ball_dets) > 0:
        bc = ball_dets.get_anchors_coordinates(sv.Position.CENTER)[0]
        cv2.circle(annotated, (int(bc[0]), int(bc[1])), 8, (0, 165, 255), -1)
        cv2.circle(annotated, (int(bc[0]), int(bc[1])), 9, (255, 255, 255), 2)

    board_img = board_drawer.draw_state(players_frame, mapped_ball)
    return annotated, board_img
