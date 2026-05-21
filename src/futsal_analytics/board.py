"""Overhead tactical board renderer."""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class TacticalBoard:
    """Renders an overhead diagram of the futsal pitch with players and ball."""

    TEAM_COLORS: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
        (255, 80, 80),
        (80, 255, 255),
    )
    PITCH_COLOR: Tuple[int, int, int] = (34, 139, 34)

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def _draw_pitch(self, board: np.ndarray) -> None:
        """Draw standard futsal pitch markings."""
        line_color = (255, 255, 255)
        cx = self.width // 2
        cy = self.height // 2

        cv2.rectangle(board, (0, 0), (self.width - 1, self.height - 1), line_color, 3)
        cv2.line(board, (cx, 0), (cx, self.height), line_color, 2)
        cv2.circle(board, (cx, cy), 5, line_color, -1)
        cv2.circle(board, (cx, cy), 60, line_color, 2)

        pen_box_w, pen_box_h = 120, 200
        goal_box_w, goal_box_h = 60, 100

        left_pen_y1 = cy - pen_box_h // 2
        left_pen_y2 = cy + pen_box_h // 2
        left_goal_y1 = cy - goal_box_h // 2
        left_goal_y2 = cy + goal_box_h // 2

        cv2.rectangle(board, (0, left_pen_y1), (pen_box_w, left_pen_y2), line_color, 2)
        cv2.rectangle(board, (0, left_goal_y1), (goal_box_w, left_goal_y2), line_color, 2)
        cv2.circle(board, (int(pen_box_w * 0.6), cy), 3, line_color, -1)

        right_pen_y1 = cy - pen_box_h // 2
        right_pen_y2 = cy + pen_box_h // 2
        right_goal_y1 = cy - goal_box_h // 2
        right_goal_y2 = cy + goal_box_h // 2

        cv2.rectangle(
            board,
            (self.width - pen_box_w, right_pen_y1),
            (self.width, right_pen_y2),
            line_color,
            2,
        )
        cv2.rectangle(
            board,
            (self.width - goal_box_w, right_goal_y1),
            (self.width, right_goal_y2),
            line_color,
            2,
        )
        cv2.circle(board, (self.width - int(pen_box_w * 0.6), cy), 3, line_color, -1)

    def draw_state(
        self,
        players: List[Tuple[int, np.ndarray, int]],
        ball_mapped: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Render the current match state.

        Args:
            players: List of (track_id, board_position, team_id).
                     team_id == -1 means referee (not drawn).
            ball_mapped: Ball position in board coordinates, or None.

        Returns:
            BGR image of the rendered board.
        """
        board = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        board[:] = self.PITCH_COLOR
        self._draw_pitch(board)

        for _track_id, pos, team_id in players:
            if team_id < 0:
                continue
            x = int(np.clip(pos[0], 5, self.width - 5))
            y = int(np.clip(pos[1], 5, self.height - 5))
            color = self.TEAM_COLORS[team_id % len(self.TEAM_COLORS)]
            cv2.circle(board, (x, y), 12, color, -1)
            cv2.circle(board, (x, y), 12, (255, 255, 255), 2)

        if ball_mapped is not None:
            bx = int(np.clip(ball_mapped[0], 3, self.width - 3))
            by = int(np.clip(ball_mapped[1], 3, self.height - 3))
            cv2.circle(board, (bx, by), 7, (0, 165, 255), -1)
            cv2.circle(board, (bx, by), 8, (255, 255, 255), 1)

        return board
