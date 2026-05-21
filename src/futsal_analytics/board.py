"""Overhead tactical board renderer (futsal-correct pitch markings)."""

from typing import List, Optional, Tuple

import cv2
import numpy as np

# Real futsal pitch dimensions (FIFA international standard, in metres).
PITCH_WIDTH_M = 40.0
PITCH_HEIGHT_M = 20.0


class TacticalBoard:
    """
    Renders an overhead diagram of a futsal pitch with players and ball.

    Pitch markings follow the FIFA futsal laws of the game:
        - 40 m × 20 m playing area (board scaled to this aspect ratio in metres)
        - Halfway line + 3 m centre circle
        - D-shaped penalty areas: two 6 m quarter-circles + a 3.16 m line
        - First penalty mark at 6 m, second penalty mark at 10 m
        - Goals: 3 m wide
    """

    TEAM_COLORS: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
        (255, 80, 80),    # team 0 — blue-ish
        (80, 255, 255),   # team 1 — yellow-ish
    )
    REFEREE_COLOR: Tuple[int, int, int] = (200, 200, 200)
    BALL_COLOR: Tuple[int, int, int] = (0, 165, 255)
    PITCH_COLOR: Tuple[int, int, int] = (34, 139, 34)
    LINE_COLOR: Tuple[int, int, int] = (255, 255, 255)

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.m_per_px_x = PITCH_WIDTH_M / width
        self.m_per_px_y = PITCH_HEIGHT_M / height

    @property
    def px_per_m_x(self) -> float:
        return self.width / PITCH_WIDTH_M

    @property
    def px_per_m_y(self) -> float:
        return self.height / PITCH_HEIGHT_M

    # ------------------------------------------------------------------
    # Pitch drawing
    # ------------------------------------------------------------------

    def _draw_pitch(self, board: np.ndarray) -> None:
        """Render FIFA-spec futsal pitch markings."""
        line = self.LINE_COLOR
        W, H = self.width, self.height
        cx, cy = W // 2, H // 2

        px_x = self.px_per_m_x
        px_y = self.px_per_m_y

        # Outer boundary
        cv2.rectangle(board, (0, 0), (W - 1, H - 1), line, 3)

        # Halfway line + centre circle (3 m radius) + kickoff spot
        cv2.line(board, (cx, 0), (cx, H), line, 2)
        cv2.circle(board, (cx, cy), int(3.0 * px_x), line, 2)
        cv2.circle(board, (cx, cy), 3, line, -1)

        # Goal-mouth and penalty area parameters
        goal_half_m = 1.5            # goal is 3 m wide
        penalty_radius_m = 6.0       # quarter-circles forming the D
        first_pk_m = 6.0
        second_pk_m = 10.0

        goal_half_px = int(goal_half_m * px_y)
        r_px = int(penalty_radius_m * px_x)
        top_post_y = cy - goal_half_px
        bot_post_y = cy + goal_half_px

        # Left D-shaped penalty area
        # Quarter-circle from top post (angles 0..90 degrees → right + down)
        cv2.ellipse(board, (0, top_post_y), (r_px, r_px), 0, 0, 90, line, 2)
        # Quarter-circle from bottom post (angles 270..360 → right + up)
        cv2.ellipse(board, (0, bot_post_y), (r_px, r_px), 0, 270, 360, line, 2)
        # Straight segment connecting the two arcs at x = r_px
        cv2.line(board, (r_px, top_post_y), (r_px, bot_post_y), line, 2)
        # Penalty marks
        cv2.circle(board, (int(first_pk_m * px_x), cy), 3, line, -1)
        cv2.circle(board, (int(second_pk_m * px_x), cy), 3, line, -1)
        # Goal posts
        cv2.line(board, (0, top_post_y), (0, bot_post_y), (0, 0, 0), 4)

        # Right D-shaped penalty area (mirror)
        cv2.ellipse(board, (W, top_post_y), (r_px, r_px), 0, 90, 180, line, 2)
        cv2.ellipse(board, (W, bot_post_y), (r_px, r_px), 0, 180, 270, line, 2)
        cv2.line(board, (W - r_px, top_post_y), (W - r_px, bot_post_y), line, 2)
        cv2.circle(board, (W - int(first_pk_m * px_x), cy), 3, line, -1)
        cv2.circle(board, (W - int(second_pk_m * px_x), cy), 3, line, -1)
        cv2.line(board, (W - 1, top_post_y), (W - 1, bot_post_y), (0, 0, 0), 4)

    # ------------------------------------------------------------------
    # Drawing players / ball
    # ------------------------------------------------------------------

    def draw_state(
        self,
        players: List[Tuple[int, np.ndarray, int]],
        ball_mapped: Optional[np.ndarray],
        show_ids: bool = True,
    ) -> np.ndarray:
        """
        Render the current match state.

        Args:
            players: List of (track_id, board_position, team_id). team_id == -1
                     is drawn in grey (referee).
            ball_mapped: Ball position in board coordinates, or None.
            show_ids: If True, draw the track_id over each player marker.

        Returns:
            BGR image of the rendered board.
        """
        board = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        board[:] = self.PITCH_COLOR
        self._draw_pitch(board)

        for track_id, pos, team_id in players:
            x = int(np.clip(pos[0], 5, self.width - 5))
            y = int(np.clip(pos[1], 5, self.height - 5))

            if team_id < 0:
                color = self.REFEREE_COLOR
            else:
                color = self.TEAM_COLORS[team_id % len(self.TEAM_COLORS)]

            cv2.circle(board, (x, y), 12, color, -1)
            cv2.circle(board, (x, y), 12, (255, 255, 255), 2)

            if show_ids and track_id > 0:
                cv2.putText(
                    board,
                    str(track_id),
                    (x - 8, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        if ball_mapped is not None:
            bx = int(np.clip(ball_mapped[0], 3, self.width - 3))
            by = int(np.clip(ball_mapped[1], 3, self.height - 3))
            cv2.circle(board, (bx, by), 7, self.BALL_COLOR, -1)
            cv2.circle(board, (bx, by), 8, (255, 255, 255), 1)

        return board
