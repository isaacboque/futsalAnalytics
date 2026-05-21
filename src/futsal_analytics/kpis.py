"""
Per-player KPI accumulation: distance covered, top speed, sprints,
possession, and duel events.

Coordinates are expected in **tactical-board pixel space**, and conversion to
metres uses the board's known real-world mapping (40 m × 20 m futsal pitch).
"""

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from futsal_analytics.board import PITCH_HEIGHT_M, PITCH_WIDTH_M

logger = logging.getLogger(__name__)


@dataclass
class PlayerStats:
    """Accumulated per-player KPIs."""
    track_id: int
    team: int = -1
    distance_m: float = 0.0
    top_speed_ms: float = 0.0
    sprint_count: int = 0
    duel_frames: int = 0
    possession_frames: int = 0
    seen_frames: int = 0
    last_pos: Optional[np.ndarray] = field(default=None, repr=False)
    last_time: Optional[float] = field(default=None, repr=False)
    in_sprint: bool = field(default=False, repr=False)


class KPITracker:
    """
    Accumulates per-player KPIs across the duration of a match analysis.

    Args:
        fps: Playback frame rate (used to convert frame counts to seconds).
        board_width_px: Width of the tactical board in pixels.
        board_height_px: Height of the tactical board in pixels.
        sprint_threshold_ms: Speed threshold (m/s) to count as a sprint.
            Futsal sprints typically start around 4–5 m/s.
        duel_radius_m: Distance under which two opposing players are "in a duel".
        possession_radius_m: Distance from the ball under which a player is
            considered to have possession.
    """

    def __init__(
        self,
        fps: float,
        board_width_px: int,
        board_height_px: int,
        sprint_threshold_ms: float = 5.0,
        duel_radius_m: float = 1.5,
        possession_radius_m: float = 1.5,
    ) -> None:
        self.fps = fps if fps > 0 else 25.0
        # Per-axis pixel-to-metre scale. Each pixel displacement (dx, dy) is
        # converted to metres component-wise (dx * m_per_px_x, dy * m_per_px_y)
        # before computing Euclidean distance. Averaging the two scales — as
        # the previous implementation did — biases distance estimates whenever
        # the tactical board's aspect ratio doesn't match the 2:1 pitch ratio.
        self.m_per_px_x = PITCH_WIDTH_M / board_width_px
        self.m_per_px_y = PITCH_HEIGHT_M / board_height_px
        self.sprint_threshold_ms = sprint_threshold_ms
        self.duel_radius_m = duel_radius_m
        self.possession_radius_m = possession_radius_m

        self.stats: Dict[int, PlayerStats] = {}
        self.total_frames: int = 0

    # ------------------------------------------------------------------

    def _to_metres(self, pos_px: np.ndarray) -> np.ndarray:
        """Convert board-pixel coordinates to metres."""
        return np.array([pos_px[0] * self.m_per_px_x, pos_px[1] * self.m_per_px_y])

    def _player_stats(self, track_id: int) -> PlayerStats:
        if track_id not in self.stats:
            self.stats[track_id] = PlayerStats(track_id=track_id)
        return self.stats[track_id]

    # ------------------------------------------------------------------

    def update(
        self,
        frame_idx: int,
        players: List[Tuple[int, np.ndarray, int]],
        ball_pos: Optional[np.ndarray],
    ) -> None:
        """
        Update KPIs for one frame.

        Args:
            frame_idx: Sequential frame number (0-based).
            players: List of (track_id, board_pos_px, team_id) tuples.
            ball_pos: Ball position in board pixels, or None.
        """
        self.total_frames = max(self.total_frames, frame_idx + 1)
        t = frame_idx / self.fps

        # Closest player to the ball (for possession)
        possessor_id: Optional[int] = None
        if ball_pos is not None and players:
            ball_m = self._to_metres(ball_pos)
            nearest = float("inf")
            for tid, pos, _ in players:
                if tid <= 0:
                    continue
                d_m = float(np.linalg.norm(self._to_metres(pos) - ball_m))
                if d_m < nearest and d_m < self.possession_radius_m:
                    nearest = d_m
                    possessor_id = tid

        # Per-player updates
        for tid, pos, team_id in players:
            if tid <= 0:
                continue  # skip detections without a real track ID
            st = self._player_stats(tid)
            st.team = team_id
            st.seen_frames += 1

            pos_m = self._to_metres(pos)

            if st.last_pos is not None and st.last_time is not None:
                step_m = float(np.linalg.norm(pos_m - st.last_pos))
                dt = t - st.last_time
                if dt > 0:
                    speed = step_m / dt
                    # 10 m/s ≈ 36 km/h — already faster than the world-record
                    # 100 m sprint average. Anything above is almost certainly
                    # a track-ID swap teleport, so drop the sample entirely.
                    if speed < 10.0:
                        st.distance_m += step_m
                        st.top_speed_ms = max(st.top_speed_ms, speed)
                        if speed >= self.sprint_threshold_ms:
                            if not st.in_sprint:
                                st.sprint_count += 1
                                st.in_sprint = True
                        else:
                            st.in_sprint = False
                    else:
                        st.in_sprint = False

            st.last_pos = pos_m
            st.last_time = t

            if tid == possessor_id:
                st.possession_frames += 1

        # Duel detection: opposing-team players within duel_radius_m
        valid_players = [(tid, self._to_metres(pos), team) for tid, pos, team in players if tid > 0]
        for i, (tid_a, pos_a, team_a) in enumerate(valid_players):
            for tid_b, pos_b, team_b in valid_players[i + 1:]:
                if team_a == team_b or team_a < 0 or team_b < 0:
                    continue
                if float(np.linalg.norm(pos_a - pos_b)) < self.duel_radius_m:
                    self.stats[tid_a].duel_frames += 1
                    self.stats[tid_b].duel_frames += 1

    # ------------------------------------------------------------------

    def to_rows(self) -> List[Dict[str, float]]:
        """Return one dict per tracked player, sorted by track_id."""
        rows = []
        for tid in sorted(self.stats.keys()):
            st = self.stats[tid]
            rows.append(
                {
                    "track_id": tid,
                    "team": st.team,
                    "distance_m": round(st.distance_m, 1),
                    "top_speed_ms": round(st.top_speed_ms, 2),
                    "sprint_count": st.sprint_count,
                    "possession_s": round(st.possession_frames / self.fps, 2),
                    "duel_s": round(st.duel_frames / self.fps, 2),
                    "seen_s": round(st.seen_frames / self.fps, 1),
                }
            )
        return rows

    def save_csv(self, path: Path, *, quiet: bool = False) -> None:
        """Write per-player KPIs to a CSV file.

        Writes atomically (``.tmp`` then ``os.replace``) so a Streamlit viewer
        reading mid-flush sees either the previous full file or the new full
        file, never a half-written one. Safe to call from the main loop every
        few frames for live dashboards.

        Args:
            path: Destination CSV path.
            quiet: When True, suppress the "wrote N rows" info log. The main
                   loop uses this for periodic incremental flushes to avoid
                   filling logs with one INFO line per flush.
        """
        import os

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_rows()
        if not rows:
            if not quiet:
                logger.warning("No KPI rows to write to %s", path)
            return
        tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
        with tmp.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
        if not quiet:
            logger.info("[KPI] Wrote %d player rows to %s", len(rows), path)

    def summary(self) -> str:
        """Return a multi-line human-readable summary."""
        rows = self.to_rows()
        if not rows:
            return "No KPI data collected."

        lines = [
            f"{'ID':>4} {'TEAM':>5} {'DIST(m)':>8} {'TOP(m/s)':>9} "
            f"{'SPRINTS':>8} {'POSS(s)':>8} {'DUEL(s)':>8} {'SEEN(s)':>8}"
        ]
        for r in rows:
            lines.append(
                f"{r['track_id']:>4} {r['team']:>5} {r['distance_m']:>8.1f} "
                f"{r['top_speed_ms']:>9.2f} {r['sprint_count']:>8} "
                f"{r['possession_s']:>8.2f} {r['duel_s']:>8.2f} {r['seen_s']:>8.1f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSONL helper for per-frame position dumps
# ---------------------------------------------------------------------------


class PositionLogger:
    """Append per-frame player and ball positions as JSON Lines."""

    def __init__(self, path: Path) -> None:
        import json

        self._json = json
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("w", encoding="utf-8")

    def log(
        self,
        frame_idx: int,
        timestamp_s: float,
        players: List[Tuple[int, np.ndarray, int]],
        ball_pos: Optional[np.ndarray],
    ) -> None:
        record = {
            "frame": frame_idx,
            "t": round(timestamp_s, 3),
            "players": [
                {
                    "id": int(tid),
                    "team": int(team),
                    "x": round(float(pos[0]), 2),
                    "y": round(float(pos[1]), 2),
                }
                for tid, pos, team in players
                if tid > 0
            ],
            "ball": (
                {"x": round(float(ball_pos[0]), 2), "y": round(float(ball_pos[1]), 2)}
                if ball_pos is not None
                else None
            ),
        }
        self._fp.write(self._json.dumps(record) + "\n")

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()
            logger.info("[POSITIONS] Wrote per-frame log to %s", self.path)
