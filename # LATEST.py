"""
Football Match Analysis System
Extracts tactical and performance metrics from YouTube stream videos using YOLO detection,
team classification, perspective transformation, and KPI computation.
"""

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import defaultdict, deque
import csv
import time
import os
import subprocess
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# ===================== LOGGING SETUP =====================

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ===================== CONFIGURATION CLASS =====================

@dataclass
class Config:
    """Centralized configuration for the analysis system."""
    model_name: str = "yolo11n.pt"
    board_width: int = 800
    board_height: int = 400
    max_display_width: int = 1280
    num_calib_points: int = 6
    
    player_class_id: int = 0
    ball_class_id: int = 32
    
    min_players_for_train: int = 8
    min_players_for_kmeans: int = 6
    
    smoothing_window: int = 5       # Frames for position smoothing
    duel_radius_px: int = 40        # Board-pixels: duel proximity threshold
    possession_radius_px: int = 50  # Board-pixels: ball possession threshold
    
    yolo_conf_threshold: float = 0.3
    track_activation_threshold: float = 0.25
    
    output_csv: str = "match_data.csv"
    stream_read_retries: int = 30
    stream_read_delay: float = 0.3
    yt_dlp_timeout: int = 30
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.board_width <= 0 or self.board_height <= 0:
            raise ValueError("Board dimensions must be positive")
        if self.smoothing_window < 1:
            raise ValueError("Smoothing window must be >= 1")


# For backward compatibility
config = Config()


# ===================== YOUTUBE STREAM =====================

def open_youtube_stream(url: str, config: Config) -> Optional[cv2.VideoCapture]:
    """
    Opens a YouTube video as an OpenCV VideoCapture by extracting the direct
    stream URL via yt-dlp. Tries to get the best available quality first,
    then falls back to lower qualities if needed.
    
    Args:
        url: YouTube video URL
        config: Configuration object with timeout settings
        
    Returns:
        cv2.VideoCapture if successful, None otherwise.
    """
    formats = ["best", "best[height<=1080]", "best[height<=720]", "best[height<=480]", "best[height<=360]"]
    
    for fmt in formats:
        try:
            result = subprocess.run(
                ["yt-dlp", "-f", fmt, "--get-url", url],
                capture_output=True,
                text=True,
                timeout=config.yt_dlp_timeout
            )
            
            if result.returncode != 0:
                logger.warning(f"yt-dlp failed for {fmt}: {result.stderr.strip()[:120]}")
                continue
            
            stream_url = result.stdout.strip().split("\n")[0]
            if not stream_url:
                continue
            
            cap = cv2.VideoCapture(stream_url)
            if cap.isOpened():
                logger.info(f"Stream opened successfully ({fmt})")
                return cap
            cap.release()
            
        except FileNotFoundError:
            logger.error("yt-dlp not found. Install with: pip install yt-dlp")
            return None
        except subprocess.TimeoutExpired:
            logger.warning(f"yt-dlp timeout for {fmt}")
    
    logger.error("Failed to open stream with any format")
    return None


# ===================== TEAM CLASSIFIER =====================

# ===================== FIELD VALIDATOR =====================

class FieldValidator:
    """
    Validates if detected players are within the playing field boundaries.
    Uses convex hull of calibration points to define valid region.
    """

    def __init__(self, src_pts: np.ndarray):
        """
        Initialize field validator with calibration points.
        
        Args:
            src_pts: Array of 6 calibration points (camera space)
        """
        if len(src_pts) != 6:
            raise ValueError("Need exactly 6 calibration points")
        
        # Create convex hull from the 4 corner points (ignore midline points initially)
        tl, tr, br, bl, mt, mb = src_pts
        corners = np.array([tl, tr, br, bl], dtype=np.float32)
        
        # Calculate convex hull that encompasses the field
        hull = cv2.convexHull(corners)
        self.field_polygon = hull
        logger.info(f"Field validator initialized with polygon: {hull.shape}")

    def is_within_field(self, point: np.ndarray) -> bool:
        """
        Check if a point is within the field boundaries.
        
        Args:
            point: Point [x, y] in camera space
            
        Returns:
            True if point is within field polygon
        """
        point_array = np.array([[[point[0], point[1]]]], dtype=np.float32)
        result = cv2.pointPolygonTest(self.field_polygon, point[:2], False)
        return result >= 0

    def filter_detections(self, detections: sv.Detections, feet_coords: np.ndarray) -> np.ndarray:
        """
        Filter detections to keep only those within the field.
        
        Args:
            detections: Supervision detections
            feet_coords: Feet coordinates (anchors) for each detection
            
        Returns:
            Boolean mask of valid detections
        """
        valid_mask = np.array([self.is_within_field(foot) for foot in feet_coords])
        return valid_mask


# ===================== TEAM CLASSIFIER =====================

class TeamClassifier:
    """
    Separates players into two teams using K-Means clustering on jersey colour.

    Uses HSV colour space with enhanced feature extraction:
    - Hue: Primary color identifier
    - Saturation: Excludes reflections and shadows
    - Value: Brightness normalization
    
    Uses 3 clusters to account for referee (smallest cluster = -1 team).
    
    Attributes:
        kmeans: KMeans clusterer instance
        trained: Whether classification model has been fitted
        ref_label: Cluster label assigned to the referee
        color_features: Extracted HSV features for analysis
    """

    def __init__(self, n_clusters: int = 3, n_init: int = 20):
        """Initialize the classifier with enhanced hyperparameters."""
        self.kmeans: KMeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
        self.trained: bool = False
        self.ref_label: Optional[int] = None
        self.color_features: Optional[np.ndarray] = None

    @staticmethod
    def _safe_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Safely crop a region from frame with boundary clipping."""
        h, w = frame.shape[:2]
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=frame.dtype)
        return frame[y1:y2, x1:x2]

    def get_jersey_color_features(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Extract enhanced HSV features from jersey region.
        Uses upper 40% of bounding box (shirt area) and filters by saturation.
        
        Args:
            frame: Input image frame
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Feature vector [hue, saturation, value] as float32
        """
        x1, y1, x2, y2 = map(int, bbox)
        jersey_h = int((y2 - y1) * 0.4)
        crop = self._safe_crop(frame, x1, y1, x2, y1 + jersey_h)
        
        if crop.size == 0:
            return np.array([0, 0, 0], dtype=np.float32)
        
        # Convert to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Filter pixels by saturation (exclude very desaturated pixels = reflections)
        saturation_mask = hsv[:, :, 1] > 30  # Min saturation threshold
        
        if saturation_mask.sum() < 5:
            # Fallback if too few saturated pixels
            return np.mean(hsv, axis=(0, 1)).astype(np.float32)
        
        # Calculate mean of saturated pixels only
        filtered_hsv = hsv[saturation_mask]
        features = np.mean(filtered_hsv, axis=0).astype(np.float32)
        
        # Normalize hue to [0, 180] range
        features[0] = np.clip(features[0], 0, 180)
        
        return features

    def train_teams(self, frame: np.ndarray, detections: sv.Detections, config: Config) -> bool:
        """
        Fit K-Means clustering on all detected players' jersey colours.
        Referee cluster is identified as the smallest population.
        
        Args:
            frame: Input image frame
            detections: Supervision detections object
            config: Configuration object
            
        Returns:
            True if training succeeded, False otherwise
        """
        if len(detections) < config.min_players_for_kmeans:
            logger.debug(f"Not enough players ({len(detections)}) for training")
            return False
        
        try:
            # Extract enhanced color features
            color_features = np.array(
                [self.get_jersey_color_features(frame, b) for b in detections.xyxy],
                dtype=np.float32
            )
            
            self.color_features = color_features
            
            # Check color variance
            hue_var = np.var(color_features[:, 0])
            sat_var = np.var(color_features[:, 1])
            
            if hue_var < 50:
                logger.warning("Insufficient color variance for team classification")
                return False
            
            # Fit K-Means
            self.kmeans.fit(color_features)
            self.trained = True
            
            # Assign referee as smallest cluster
            counts = np.bincount(self.kmeans.labels_)
            self.ref_label = int(np.argmin(counts))
            
            logger.info(
                f"Team classifier trained successfully. "
                f"Referee cluster: {self.ref_label} | "
                f"Cluster sizes: {counts} | "
                f"Hue variance: {hue_var:.1f} | "
                f"Saturation variance: {sat_var:.1f}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error during team training: {e}")
            return False

    def predict_team(self, frame: np.ndarray, bbox: np.ndarray) -> int:
        """
        Predict team ID for a player's bounding box.
        
        Args:
            frame: Input image frame
            bbox: Player bounding box
            
        Returns:
            0 or 1 for team players, -1 for referee
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
            
        except Exception as e:
            logger.warning(f"Error predicting team: {e}")
            return 0


# ===================== POSITION SMOOTHER =====================

class PositionSmoother:
    """
    Rolling average over the last N frames per player.
    Eliminates jitter from frame-to-frame detection noise on the tactical board.
    
    Attributes:
        buffers: Dictionary mapping track_id to deque of positions
    """

    def __init__(self, window: int = 5):
        """
        Initialize with smoothing window size.
        
        Args:
            window: Number of frames to average over
        """
        self.buffers: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window))

    def update(self, track_id: int, pos: np.ndarray) -> np.ndarray:
        """
        Add position for a player and return smoothed position.
        
        Args:
            track_id: Unique player identifier
            pos: Position vector [x, y]
            
        Returns:
            Smoothed position (rolling average)
        """
        self.buffers[track_id].append(pos)
        return np.mean(self.buffers[track_id], axis=0)


# ===================== KPI TRACKER =====================

# ===================== KPI TRACKER =====================

class KPITracker:
    """
    Accumulates per-player KPIs across all processed frames:

    - Distance covered  (board-pixel displacement converted to metres)
    - Duels             (frames where two opposing players are within DUEL_RADIUS_PX)
    - Possession time   (seconds as the player closest to the ball)
    - Sprint count      (frames where speed exceeds ~7 m/s threshold)

    Scale: 800px board = 40m real pitch width  =>  1px = 0.05m
    
    Attributes:
        distance: Cumulative distance per player (metres)
        duels: Count of frames in duels per player
        possession: Count of possession frames per player
        sprints: Count of sprint events per player
        prev_pos: Previous position per player for delta calculation
        team_map: Team assignment per player
    """

    BOARD_TO_METRES: float = 40.0 / 800.0  # metres per board-pixel

    def __init__(self):
        """Initialize KPI tracking dictionaries."""
        self.distance: Dict[int, float] = defaultdict(float)
        self.duels: Dict[int, int] = defaultdict(int)
        self.possession: Dict[int, int] = defaultdict(int)
        self.sprints: Dict[int, int] = defaultdict(int)
        self.prev_pos: Dict[int, np.ndarray] = {}
        self.team_map: Dict[int, int] = {}

    def update(self, players: List[Tuple[int, np.ndarray, int]], 
               ball_pos: Optional[np.ndarray], 
               fps: float,
               config: Config) -> None:
        """
        Update KPI accumulators with frame data.
        
        Args:
            players: List of (track_id, position, team_id) tuples
            ball_pos: Ball position or None
            fps: Frames per second of video
            config: Configuration object with radius parameters
        """
        if fps <= 0:
            logger.warning("Invalid FPS value")
            return
        
        # Sprints threshold: ~7 m/s
        sprint_px = (7.0 / self.BOARD_TO_METRES) / fps

        # Track distance and sprints
        for track_id, pos, team_id in players:
            self.team_map[track_id] = team_id
            
            if track_id in self.prev_pos:
                d = float(np.linalg.norm(pos - self.prev_pos[track_id]))
                self.distance[track_id] += d * self.BOARD_TO_METRES
                
                if d > sprint_px:
                    self.sprints[track_id] += 1
            
            self.prev_pos[track_id] = pos.copy()

        # Track duels (opposing players within radius)
        for i, (id_a, pos_a, team_a) in enumerate(players):
            for id_b, pos_b, team_b in players[i + 1:]:
                if team_a >= 0 and team_b >= 0 and team_a != team_b:
                    if np.linalg.norm(pos_a - pos_b) < config.duel_radius_px:
                        self.duels[id_a] += 1
                        self.duels[id_b] += 1

        # Track possession (closest player to ball)
        if ball_pos is not None and players:
            try:
                closest = min(players, key=lambda p: np.linalg.norm(p[1] - ball_pos))
                if np.linalg.norm(closest[1] - ball_pos) < config.possession_radius_px:
                    self.possession[closest[0]] += 1
            except (ValueError, IndexError):
                pass

    def get_summary(self, fps: float) -> Dict[int, Dict[str, Any]]:
        """
        Generate KPI summary dictionary for all tracked players.
        
        Args:
            fps: Frames per second for converting frames to seconds
            
        Returns:
            Dictionary mapping track_id to KPI dictionary
        """
        return {
            tid: {
                "team": self.team_map.get(tid, -1),
                "distance_m": round(self.distance[tid], 2),
                "duel_frames": self.duels[tid],
                "possession_s": round(self.possession[tid] / max(fps, 1), 2),
                "sprint_count": self.sprints[tid],
            }
            for tid in self.team_map
        }

    def export_csv(self, path: str, fps: float) -> bool:
        """
        Export KPI summary to CSV file.
        
        Args:
            path: Output file path
            fps: Frames per second for time conversion
            
        Returns:
            True if export succeeded, False otherwise
        """
        try:
            summary = self.get_summary(fps)
            if not summary:
                logger.warning("No KPI data to export")
                return False
            
            fields = ["track_id", "team", "distance_m", "duel_frames", "possession_s", "sprint_count"]
            
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for tid, kpis in summary.items():
                    w.writerow({"track_id": tid, **kpis})
            
            logger.info(f"KPIs exported to: {os.path.abspath(path)}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting KPI CSV: {e}")
            return False


# ===================== PERSPECTIVE TRANSFORMER =====================

# ===================== PERSPECTIVE TRANSFORMER =====================

class PiecewiseTransformer:
    """
    Maps camera pixel coordinates to a flat 2D board using two homographies:
    one for the left half of the pitch, one for the right.

    A single homography cannot accurately correct the non-uniform perspective
    distortion of a wide-angle fixed camera. Splitting at the halfway line and
    computing independent transforms for each half significantly improves
    positional accuracy near the edges of the frame.
    
    Attributes:
        m_left: Homography matrix for left half of pitch
        m_right: Homography matrix for right half of pitch
        midline_x: X-coordinate of field midline in camera space
    """

    def __init__(self, src_pts: np.ndarray, board_width: int, board_height: int):
        """
        Initialize perspective transformer.
        
        Args:
            src_pts: Array of 6 calibration points:
                    [top_left, top_right, bottom_right, bottom_left, 
                     midline_top, midline_bottom]
            board_width: Tactical board width in pixels
            board_height: Tactical board height in pixels
        """
        if len(src_pts) != 6:
            raise ValueError("Need exactly 6 calibration points")
        
        tl, tr, br, bl, mt, mb = src_pts
        mid = board_width // 2
        
        # Left half: TL -> MT -> MB -> BL
        self.m_left = cv2.getPerspectiveTransform(
            np.array([tl, mt, mb, bl], dtype=np.float32),
            np.array([[0, 0], [mid, 0], [mid, board_height], [0, board_height]], dtype=np.float32)
        )
        
        # Right half: MT -> TR -> BR -> MB
        self.m_right = cv2.getPerspectiveTransform(
            np.array([mt, tr, br, mb], dtype=np.float32),
            np.array([[mid, 0], [board_width, 0], [board_width, board_height], [mid, board_height]], dtype=np.float32)
        )
        
        self.midline_x: float = float((mt[0] + mb[0]) / 2.0)
        logger.info(f"Perspective transformer initialized (midline at x={self.midline_x:.1f})")

    def transform(self, pt: np.ndarray) -> np.ndarray:
        """
        Transform a camera-space point to board-space coordinates.
        
        Args:
            pt: Point [x, y] in camera space
            
        Returns:
            Transformed point [x, y] in board space
        """
        if pt[0] < 0 or pt[1] < 0:
            return pt
        
        mat = self.m_left if pt[0] < self.midline_x else self.m_right
        return cv2.perspectiveTransform(np.array([[pt]], dtype=np.float32), mat)[0][0]


# ===================== TACTICAL BOARD =====================

# ===================== TACTICAL BOARD =====================

class TacticalBoard:
    """
    Renders a 2D overhead pitch diagram with players, ball, and KPI overlay.
    
    Attributes:
        width: Board width in pixels
        height: Board height in pixels
        team_colors: Color tuple for each team in BGR format
    """

    # Team colors (BGR): Blue, Yellow
    TEAM_COLORS: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = ((255, 80, 80), (80, 255, 255))
    PITCH_COLOR: Tuple[int, int, int] = (34, 139, 34)  # Dark green

    def __init__(self, width: int, height: int):
        """Initialize tactical board with given dimensions."""
        self.width: int = width
        self.height: int = height

    def _draw_pitch(self, board: np.ndarray) -> None:
        """Draw professional pitch markings on the tactical board."""
        line_color = (255, 255, 255)
        center_x = self.width // 2
        center_y = self.height // 2
        
        # Pitch boundary (outer touchlines and goal lines) - thick lines
        cv2.rectangle(board, (0, 0), (self.width - 1, self.height - 1), line_color, 3)
        
        # Halfway line (vertical center line)
        cv2.line(board, (center_x, 0), (center_x, self.height), line_color, 2)
        
        # Center spot (midfield)
        cv2.circle(board, (center_x, center_y), 5, line_color, -1)
        
        # Center circle
        cv2.circle(board, (center_x, center_y), 75, line_color, 2)
        
        # Penalty areas dimensions (professional proportions)
        pen_box_w, pen_box_h = 150, 240
        goal_box_w, goal_box_h = 50, 90
        
        # Left side boxes
        left_pen_y1 = center_y - pen_box_h // 2
        left_pen_y2 = center_y + pen_box_h // 2
        left_goal_y1 = center_y - goal_box_h // 2
        left_goal_y2 = center_y + goal_box_h // 2
        
        # Left penalty box
        cv2.rectangle(board, (0, left_pen_y1), (pen_box_w, left_pen_y2), line_color, 2)
        # Left goal area
        cv2.rectangle(board, (0, left_goal_y1), (goal_box_w, left_goal_y2), line_color, 2)
        # Left penalty spot
        cv2.circle(board, (int(pen_box_w * 0.6), center_y), 4, line_color, -1)
        
        # Right side boxes
        right_pen_y1 = center_y - pen_box_h // 2
        right_pen_y2 = center_y + pen_box_h // 2
        right_goal_y1 = center_y - goal_box_h // 2
        right_goal_y2 = center_y + goal_box_h // 2
        
        # Right penalty box
        cv2.rectangle(board, (self.width - pen_box_w, right_pen_y1), 
                      (self.width, right_pen_y2), line_color, 2)
        # Right goal area
        cv2.rectangle(board, (self.width - goal_box_w, right_goal_y1), 
                      (self.width, right_goal_y2), line_color, 2)
        # Right penalty spot
        cv2.circle(board, (self.width - int(pen_box_w * 0.6), center_y), 4, line_color, -1)
        
        # Corner arcs (quarter circles at corners)
        corner_radius = 20
        cv2.ellipse(board, (0, 0), (corner_radius, corner_radius), 0, 0, 90, line_color, 1)
        cv2.ellipse(board, (self.width, 0), (corner_radius, corner_radius), 0, 90, 180, line_color, 1)
        cv2.ellipse(board, (self.width, self.height), (corner_radius, corner_radius), 0, 180, 270, line_color, 1)
        cv2.ellipse(board, (0, self.height), (corner_radius, corner_radius), 0, 270, 360, line_color, 1)

    def draw_state(self, 
                   players: List[Tuple[int, np.ndarray, int]], 
                   ball_mapped: Optional[np.ndarray],
                   kpi_tracker: Optional['KPITracker'] = None, 
                   fps: float = 25.0) -> np.ndarray:
        """
        Draw the current match state on the tactical board.
        
        Args:
            players: List of (track_id, position, team_id) tuples
            ball_mapped: Ball position or None
            kpi_tracker: KPI tracker instance for distance overlay
            fps: Frames per second (for reference)
            
        Returns:
            Rendered board image (numpy array)
        """
        # Initialize board with pitch color
        board = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        board[:] = self.PITCH_COLOR
        
        # Draw pitch markings
        self._draw_pitch(board)

        # Draw players (team-only visualization, no IDs)
        for track_id, pos, team_id in players:
            if team_id < 0:  # Skip referee
                continue
            
            # Clip to board boundaries
            x = int(np.clip(pos[0], 5, self.width - 5))
            y = int(np.clip(pos[1], 5, self.height - 5))
            
            # Team color
            color = self.TEAM_COLORS[team_id % len(self.TEAM_COLORS)]
            
            # Draw player circle with team color
            cv2.circle(board, (x, y), 12, color, -1)
            # White outline for better visibility
            cv2.circle(board, (x, y), 12, (255, 255, 255), 2)

        # Draw ball
        if ball_mapped is not None:
            bx = int(np.clip(ball_mapped[0], 3, self.width - 3))
            by = int(np.clip(ball_mapped[1], 3, self.height - 3))
            cv2.circle(board, (bx, by), 7, (0, 165, 255), -1)
            cv2.circle(board, (bx, by), 8, (255, 255, 255), 1)

        return board


# ===================== FIELD CALIBRATION =====================

# ===================== FIELD CALIBRATION =====================

def select_points(frame: np.ndarray, config: Config) -> np.ndarray:
    """
    Shows the first video frame and lets the user click calibration points.
    Professional interface with clear visual feedback and instructions.
    
    Click order:
        1=Top-Left  2=Top-Right  3=Bottom-Right  4=Bottom-Left
        5=Halfway-Line Top  6=Halfway-Line Bottom
    
    Args:
        frame: First frame of video
        config: Configuration with display settings
        
    Returns:
        Array of points in original video resolution (shape: 6x2)
    """
    h, w = frame.shape[:2]
    scale = config.max_display_width / w if w > config.max_display_width else 1.0
    disp = cv2.resize(frame, (int(w * scale), int(h * scale))).copy()
    pts: List[List[int]] = []
    WIN = "FIELD CALIBRATION - Court Point Selection"
    
    # Point labels and colors
    point_labels = [
        "Top-Left Corner",
        "Top-Right Corner",
        "Bottom-Right Corner",
        "Bottom-Left Corner",
        "Halfway Line (Top)",
        "Halfway Line (Bottom)"
    ]
    point_colors = [
        (0, 255, 0),      # Green for corners
        (0, 255, 0),
        (0, 255, 0),
        (0, 255, 0),
        (0, 165, 255),    # Orange for halfway line
        (0, 165, 255)
    ]

    def draw_instructions(image: np.ndarray, current_step: int) -> np.ndarray:
        """Add professional instructions overlay."""
        img = image.copy()
        overlay = img.copy()
        
        # Semi-transparent dark background for text area
        cv2.rectangle(overlay, (0, 0), (disp.shape[1], 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        
        # Title
        cv2.putText(img, "COURT CALIBRATION", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Current instruction
        instruction = f"Step {current_step + 1}/6: Click on {point_labels[current_step]}"
        cv2.putText(img, instruction, (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, point_colors[current_step], 2)
        
        # Status bar at bottom
        progress_text = f"Progress: {current_step}/{config.num_calib_points}  |  Press ESC to cancel"
        cv2.putText(img, progress_text, (15, img.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return img

    def on_click(event, x, y, flags, param) -> None:
        """Mouse callback for point selection with professional feedback."""
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < config.num_calib_points:
            pts.append([int(x / scale), int(y / scale)])
            idx = len(pts) - 1
            
            # Draw point marker
            cv2.circle(disp, (x, y), 10, point_colors[idx], -1)
            cv2.circle(disp, (x, y), 11, (255, 255, 255), 2)
            
            # Draw point number in a box
            cv2.rectangle(disp, (x - 20, y - 30), (x + 20, y - 5), point_colors[idx], -1)
            cv2.putText(disp, str(idx + 1), (x - 12, y - 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Update display
            display = draw_instructions(disp, len(pts))
            cv2.imshow(WIN, display)

    # Print calibration instructions to console
    logger.info("="*70)
    logger.info(" "*15 + "FIELD CALIBRATION - PROFESSIONAL COURT SETUP")
    logger.info("="*70)
    logger.info("Click on the following court points in order:")
    logger.info("")
    logger.info("  CORNERS:")
    logger.info("    1. Top-Left Corner           (top-left corner of the field)")
    logger.info("    2. Top-Right Corner          (top-right corner of the field)")
    logger.info("    3. Bottom-Right Corner       (bottom-right corner of the field)")
    logger.info("    4. Bottom-Left Corner        (bottom-left corner of the field)")
    logger.info("")
    logger.info("  HALFWAY LINE:")
    logger.info("    5. Halfway Line (Top)        (where midline meets top boundary)")
    logger.info("    6. Halfway Line (Bottom)     (where midline meets bottom boundary)")
    logger.info("-"*70)
    logger.info("Window will close automatically after all points are selected.")
    logger.info("Press ESC to cancel calibration.")
    logger.info("="*70 + "\n")

    # Initial display with first instruction
    display = draw_instructions(disp, 0)
    cv2.imshow(WIN, display)
    cv2.setMouseCallback(WIN, on_click)

    # Wait for user to select all points or press ESC
    while len(pts) < config.num_calib_points:
        key = cv2.waitKey(50)
        if key == 27:  # ESC to abort
            logger.warning("Calibration cancelled by user")
            break

    cv2.destroyAllWindows()
    
    if len(pts) == config.num_calib_points:
        logger.info("✓ Calibration completed successfully!\n")
    
    return np.array(pts, dtype=np.float32)


# ===================== HELPER FUNCTIONS =====================

def parse_start_time(time_str: str) -> int:
    """
    Parse start time string in MM:SS or SS format.
    
    Args:
        time_str: Time string (e.g., "01:30", "90")
        
    Returns:
        Time in seconds, or 0 if invalid
    """
    try:
        parts = time_str.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        else:
            return int(parts[0])
    except (ValueError, IndexError):
        logger.warning(f"Invalid time format '{time_str}' - starting from 0")
        return 0


def read_first_frame(cap: cv2.VideoCapture, config: Config) -> Optional[np.ndarray]:
    """
    Read first frame from video capture with retry logic.
    
    Args:
        cap: Video capture object
        config: Configuration with retry settings
        
    Returns:
        First frame or None if failed
    """
    for attempt in range(config.stream_read_retries):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            logger.info(f"First frame acquired after {attempt} attempts")
            return frame
        
        if attempt < config.stream_read_retries - 1:
            logger.debug(f"Waiting for stream buffer... ({attempt + 1}/{config.stream_read_retries})")
            time.sleep(config.stream_read_delay)
    
    logger.error("Could not read first frame")
    return None


def setup_detectors(config: Config) -> Tuple[Optional[YOLO], Optional[sv.ByteTrack]]:
    """
    Initialize YOLO model and ByteTrack tracker.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (model, tracker) or (None, None) if failed
    """
    try:
        model = YOLO(config.model_name)
        logger.info(f"YOLO model '{config.model_name}' loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return None, None
    
    tracker = sv.ByteTrack(track_activation_threshold=config.track_activation_threshold)
    logger.info("ByteTrack tracker initialized")
    
    return model, tracker


def process_frame(frame: np.ndarray,
                  model: YOLO,
                  tracker: sv.ByteTrack,
                  classifier: TeamClassifier,
                  smoother: PositionSmoother,
                  transformer: PiecewiseTransformer,
                  field_validator: 'FieldValidator',
                  kpi_tracker: KPITracker,
                  board_drawer: TacticalBoard,
                  config: Config,
                  fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single frame: detect, track, classify, validate, and render.
    
    Args:
        frame: Input video frame
        model: YOLO detector
        tracker: ByteTrack tracker
        classifier: Team classifier
        smoother: Position smoother
        transformer: Perspective transformer
        field_validator: Field boundary validator
        kpi_tracker: KPI tracker
        board_drawer: Tactical board renderer
        config: Configuration object
        fps: Frames per second
        
    Returns:
        Tuple of (annotated camera frame, tactical board frame)
    """
    # 1. YOLO detection
    results = model.predict(
        frame,
        classes=[config.player_class_id, config.ball_class_id],
        conf=config.yolo_conf_threshold,
        verbose=False
    )[0]
    detections = sv.Detections.from_ultralytics(results)

    player_dets = detections[detections.class_id == config.player_class_id]
    ball_dets = detections[detections.class_id == config.ball_class_id]

    # 1.5 Filter players: keep only those within field boundaries
    if len(player_dets) > 0:
        feet_coords = player_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        valid_mask = field_validator.filter_detections(player_dets, feet_coords)
        player_dets = player_dets[valid_mask]
        logger.debug(f"Filtered {np.sum(~valid_mask)} players outside field")

    # 2. ByteTrack for persistent IDs
    tracked = tracker.update_with_detections(player_dets)

    # 3. Train classifier once enough players visible
    if not classifier.trained and len(tracked) >= config.min_players_for_train:
        classifier.train_teams(frame, tracked, config)

    # 4. Map player positions to board with smoothing
    players_frame: List[Tuple[int, np.ndarray, int]] = []
    if len(tracked) > 0:
        feet = tracked.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        for i, foot in enumerate(feet):
            tid = int(tracked.tracker_id[i])
            team_id = classifier.predict_team(frame, tracked.xyxy[i])
            mapped = transformer.transform(foot)
            smooth = smoother.update(tid, mapped)
            players_frame.append((tid, smooth, team_id))

    # 5. Map ball position
    mapped_ball: Optional[np.ndarray] = None
    if len(ball_dets) > 0:
        ball_center = ball_dets.get_anchors_coordinates(sv.Position.CENTER)[0]
        if field_validator.is_within_field(ball_center):
            mapped_ball = transformer.transform(ball_center)

    # 6. Update KPI accumulators
    kpi_tracker.update(players_frame, mapped_ball, fps, config)

    # 7. Annotate camera frame
    annotated = frame.copy()
    for i, bbox in enumerate(tracked.xyxy):
        team_id = classifier.predict_team(frame, bbox)
        color = (128, 128, 128) if team_id < 0 else TacticalBoard.TEAM_COLORS[team_id % 2]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Add team indicator text (no player ID)
        team_text = "REF" if team_id < 0 else f"TEAM {team_id}"
        cv2.putText(annotated, team_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 8. Render tactical board
    board_img = board_drawer.draw_state(players_frame, mapped_ball, kpi_tracker, fps)

    return annotated, board_img


# ===================== MAIN =====================

def main(cfg: Optional[Config] = None) -> None:
    """
    Main analysis pipeline.
    
    Args:
        cfg: Optional custom configuration (uses default if None)
    """
    cfg = cfg or Config()
    
    # Get input from user
    url = input("YouTube URL: ").strip()
    if not url:
        logger.error("Empty URL provided")
        return

    start_time_str = input("Start time (MM:SS, optional -- press Enter to skip): ").strip()
    
    # Open stream
    logger.info("Opening YouTube stream via yt-dlp...")
    cap = open_youtube_stream(url, cfg)
    if cap is None or not cap.isOpened():
        logger.error("Failed to open stream. Common fixes:")
        logger.error("  1. pip install --upgrade yt-dlp")
        logger.error("  2. Verify the video is public and not age-restricted")
        return

    # Seek to start time
    start_seconds = parse_start_time(start_time_str)
    if start_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)
        logger.info(f"Seeking to {start_seconds} seconds")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    logger.info(f"Video FPS: {fps}")

    # Read first frame
    first_frame = read_first_frame(cap, cfg)
    if first_frame is None:
        cap.release()
        return

    # Calibration
    source_points = select_points(first_frame.copy(), cfg)
    if len(source_points) != cfg.num_calib_points:
        logger.error(f"Need {cfg.num_calib_points} calibration points, got {len(source_points)}")
        cap.release()
        return

    try:
        transformer = PiecewiseTransformer(source_points, cfg.board_width, cfg.board_height)
    except Exception as e:
        logger.error(f"Perspective transform setup failed: {e}")
        cap.release()
        return

    # Initialize field validator
    try:
        field_validator = FieldValidator(source_points)
    except Exception as e:
        logger.error(f"Field validator setup failed: {e}")
        cap.release()
        return

    # Initialize components
    board_drawer = TacticalBoard(cfg.board_width, cfg.board_height)
    classifier = TeamClassifier()
    smoother = PositionSmoother(cfg.smoothing_window)
    kpi_tracker = KPITracker()

    model, tracker = setup_detectors(cfg)
    if model is None or tracker is None:
        cap.release()
        return

    # Create display windows
    cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Tactical Board", cv2.WINDOW_NORMAL)

    frame_idx = 0
    t_start = time.time()
    logger.info("Processing started - press Q to stop and export KPIs")

    try:
        # Main frame loop
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info("End of stream reached")
                break

            frame_idx += 1

            # Process frame and get visualizations
            try:
                annotated, board_img = process_frame(
                    frame, model, tracker, classifier, smoother, transformer,
                    field_validator, kpi_tracker, board_drawer, cfg, fps
                )
            except Exception as e:
                logger.error(f"Error processing frame {frame_idx}: {e}")
                continue

            # Add frame counter overlay
            elapsed = time.time() - t_start
            cv2.putText(annotated, f"Frame {frame_idx} | {elapsed:.0f}s",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display
            cv2.imshow("Camera View", annotated)
            cv2.imshow("Tactical Board", board_img)

            # Check for exit command
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Stream interrupted by user (Q pressed)")
                break

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

    # Export results
    logger.info(f"Processing completed - {frame_idx} frames analyzed")
    logger.info("=== MATCH KPI SUMMARY ===")
    
    summary = kpi_tracker.get_summary(fps)
    for tid, kpis in sorted(summary.items()):
        logger.info(
            f"  Player {tid:3d} | Team {kpis['team']} | "
            f"Distance: {kpis['distance_m']:6.1f}m | "
            f"Duels: {kpis['duel_frames']:4d} | "
            f"Possession: {kpis['possession_s']:5.1f}s | "
            f"Sprints: {kpis['sprint_count']:3d}"
        )

    kpi_tracker.export_csv(cfg.output_csv, fps)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")