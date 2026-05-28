"""Central configuration for the futsal analytics system."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Centralised settings. Override individual fields when instantiating.

    Only fields read elsewhere in the codebase live here. KPI thresholds
    (sprint speed, duel/possession radius in metres) are owned by
    :class:`futsal_analytics.kpis.KPITracker` because they are tied to its
    metres-per-pixel scale.
    """

    model_name: str = "runs/detect/runs/detect/futsal_train-7/weights/best.pt"
    board_width: int = 700
    board_height: int = 350

    player_class_id: int = 0
    ball_class_id: int = 32

    min_players_for_kmeans: int = 6

    yolo_conf_threshold: float = 0.3

    stream_read_retries: int = 30
    stream_read_delay: float = 0.3
    yt_dlp_timeout: int = 30

    def __post_init__(self) -> None:
        if not Path(self.model_name).is_absolute():
            candidate = Path(__file__).resolve().parents[2] / self.model_name
            if candidate.exists():
                self.model_name = str(candidate)

        if self.board_width <= 0 or self.board_height <= 0:
            raise ValueError("Board dimensions must be positive")
        if self.min_players_for_kmeans < 2:
            raise ValueError("Need at least 2 players to fit team clusters")
