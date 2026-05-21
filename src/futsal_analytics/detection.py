"""
Player and ball detection, team classification, persistent tracking,
and per-frame processing.

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
    """
    Assigns players to teams via K-Means clustering on HSV jersey colour.

    Can be re-trained on a new batch of detections to adapt to lighting changes.
    """

    def __init__(self, n_clusters: int = 3, n_init: int = 20) -> None:
        from sklearn.cluster import KMeans

        self._KMeans = KMeans
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
        self.trained: bool = False
        self.ref_label: Optional[int] = None
        self.last_trained_frame: int = -1

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
        """Fit K-Means on jersey colours of the currently visible players."""
        if len(detections) < config.min_players_for_kmeans:
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

            # Fresh KMeans instance each time to avoid sklearn warm-start side effects
            self.kmeans = self._KMeans(n_clusters=self.n_clusters, n_init=self.n_init, random_state=42)
            self.kmeans.fit(features)
            self.trained = True
            counts = np.bincount(self.kmeans.labels_)
            self.ref_label = int(np.argmin(counts))

            logger.info(
                "[TRAIN] K-Means trained. Referee cluster=%d, distribution=%s",
                self.ref_label,
                list(counts),
            )
            return True

        except Exception as exc:
            logger.error("[TRAIN] Error: %s", exc)
            return False

    def predict_team(self, frame: np.ndarray, bbox: np.ndarray) -> int:
        """Return 0 or 1 for team assignment, -1 for referee. Returns 0 if untrained."""
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
    """Frame-to-frame ball position smoother (not a true multi-object tracker)."""

    def __init__(self, max_distance: float = 100, history_size: int = 5) -> None:
        self.last_pos: Optional[np.ndarray] = None
        self.max_distance = max_distance
        self.history_size = history_size
        self.ball_seen_count: int = 0
        self._history: List[np.ndarray] = []

    def update(self, current_pos: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Update with the ball's detected position; returns smoothed position."""
        if current_pos is None:
            self.last_pos = None
            self._history = []
            return None

        if self.last_pos is not None:
            dist = float(np.linalg.norm(current_pos - self.last_pos))
            if dist > self.max_distance:
                logger.debug("Ball discontinuity (dist=%.1f) — resetting history", dist)
                self._history = []

        self.last_pos = current_pos.copy()
        self.ball_seen_count += 1
        self._history.append(current_pos.copy())
        if len(self._history) > self.history_size:
            self._history = self._history[-self.history_size:]

        return np.mean(self._history, axis=0)


# ---------------------------------------------------------------------------
# YOLO setup
# ---------------------------------------------------------------------------


def resolve_device(device: str = "auto") -> str:
    """Resolve 'auto' to 'cuda' if available, else 'cpu'."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001 — torch DLL/OSError on broken installs
        pass
    return "cpu"


def setup_detectors(config: Config) -> Optional[Any]:
    """Load the YOLO model specified in *config*."""
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


def make_tracker() -> Any:
    """Construct a fresh ByteTrack tracker from supervision."""
    import supervision as sv

    return sv.ByteTrack()


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
    fps: float,
    *,
    tracker: Any = None,
    device: str = "cpu",
    frame_idx: int = 0,
    retrain_every: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, np.ndarray, int]], Optional[np.ndarray]]:
    """
    Run the full per-frame detection pipeline.

    Returns:
        (annotated_camera_frame, tactical_board_image, players_with_ids, ball_board_pos)
        where ``players_with_ids`` is a list of (track_id, board_pos, team_id).
    """
    import supervision as sv

    results = model.predict(
        frame,
        classes=[config.player_class_id, config.ball_class_id],
        conf=config.yolo_conf_threshold,
        verbose=False,
        iou=0.5,
        device=device,
    )[0]

    detections = sv.Detections.from_ultralytics(results)
    player_dets = detections[detections.class_id == config.player_class_id]
    ball_dets = detections[detections.class_id == config.ball_class_id]

    # Size filter — scale by frame area to be resolution-independent.
    # Reference: at 1280×720 (921,600 px) the min area is 300 px.
    if len(player_dets) > 0:
        frame_area = frame.shape[0] * frame.shape[1]
        min_area = max(50, int(300 * frame_area / (1280 * 720)))
        areas = (player_dets.xyxy[:, 2] - player_dets.xyxy[:, 0]) * (
            player_dets.xyxy[:, 3] - player_dets.xyxy[:, 1]
        )
        player_dets = player_dets[areas > min_area]

    # Pitch polygon filter
    if len(player_dets) > 0:
        feet = player_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        valid_mask = field_validator.filter_detections(player_dets, feet)
        player_dets = player_dets[valid_mask]

    # ByteTrack: assign persistent IDs to surviving detections
    if tracker is not None and len(player_dets) > 0:
        try:
            player_dets = tracker.update_with_detections(player_dets)
        except Exception as exc:
            logger.warning("ByteTrack update failed: %s", exc)

    # (Re-)train classifier if needed
    if not classifier.trained and len(player_dets) >= config.min_players_for_kmeans:
        if classifier.train_teams(frame, player_dets, config):
            classifier.last_trained_frame = frame_idx
    elif (
        retrain_every > 0
        and classifier.trained
        and frame_idx - classifier.last_trained_frame >= retrain_every
        and len(player_dets) >= config.min_players_for_kmeans
    ):
        if classifier.train_teams(frame, player_dets, config):
            classifier.last_trained_frame = frame_idx
            logger.info("[TRAIN] Classifier retrained at frame %d", frame_idx)

    # Per-frame team-id cache: compute once per detection, reuse for mapping + annotation
    team_cache: List[int] = []
    if len(player_dets) > 0:
        team_cache = [classifier.predict_team(frame, bbox) for bbox in player_dets.xyxy]

    # Map players to board
    players_frame: List[Tuple[int, np.ndarray, int]] = []
    if len(player_dets) > 0:
        feet = player_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        # tracker_id may be None per-detection (newly created); fall back to a sentinel
        track_ids = (
            player_dets.tracker_id
            if player_dets.tracker_id is not None
            else np.arange(1, len(player_dets) + 1)
        )
        mapped_all = mapper.transform_batch(feet)
        for i, mapped in enumerate(mapped_all):
            tid_raw = track_ids[i] if i < len(track_ids) else None
            tid = int(tid_raw) if tid_raw is not None else -(i + 1)
            players_frame.append((tid, mapped, team_cache[i]))

    # Ball
    current_ball: Optional[np.ndarray] = None
    if len(ball_dets) > 0:
        ball_center = ball_dets.get_anchors_coordinates(sv.Position.CENTER)[0]
        if field_validator.is_within_field(ball_center):
            current_ball = mapper.transform(ball_center)
    mapped_ball = ball_tracker.update(current_ball)

    # Annotate camera frame using cached team IDs
    annotated = frame.copy()
    for i, bbox in enumerate(player_dets.xyxy):
        team_id = team_cache[i]
        color = TacticalBoard.REFEREE_COLOR if team_id < 0 else TacticalBoard.TEAM_COLORS[team_id % 2]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        # Draw the persistent ID over each bbox
        tid_raw = player_dets.tracker_id[i] if player_dets.tracker_id is not None else None
        if tid_raw is not None:
            cv2.putText(
                annotated,
                f"#{int(tid_raw)}",
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    if len(ball_dets) > 0:
        bc = ball_dets.get_anchors_coordinates(sv.Position.CENTER)[0]
        cv2.circle(annotated, (int(bc[0]), int(bc[1])), 8, TacticalBoard.BALL_COLOR, -1)
        cv2.circle(annotated, (int(bc[0]), int(bc[1])), 9, (255, 255, 255), 2)

    board_img = board_drawer.draw_state(players_frame, mapped_ball)
    return annotated, board_img, players_frame, mapped_ball
