"""Central configuration for the futsal analytics system."""

from dataclasses import dataclass


@dataclass
class Config:
    """Centralised settings. Override individual fields when instantiating."""

    model_name: str = "yolo11n.pt"
    board_width: int = 700
    board_height: int = 350

    player_class_id: int = 0
    ball_class_id: int = 32

    min_players_for_train: int = 8
    min_players_for_kmeans: int = 6

    smoothing_window: int = 5
    duel_radius_px: int = 40
    possession_radius_px: int = 50

    yolo_conf_threshold: float = 0.3
    track_activation_threshold: float = 0.25

    stream_read_retries: int = 30
    stream_read_delay: float = 0.3
    yt_dlp_timeout: int = 30

    def __post_init__(self) -> None:
        if self.board_width <= 0 or self.board_height <= 0:
            raise ValueError("Board dimensions must be positive")
        if self.smoothing_window < 1:
            raise ValueError("Smoothing window must be >= 1")
