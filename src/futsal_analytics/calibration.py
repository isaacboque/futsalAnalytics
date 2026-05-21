"""
Interactive 6-point field calibration UI.

The six control points define the futsal pitch polygon viewed from a fixed
side camera:

    Index  Label  Position
    -----  -----  ---------------------
      0    TL     Top-Left corner
      1    CT     Centre-Top (halfway line, top edge)
      2    TR     Top-Right corner
      3    BR     Bottom-Right corner
      4    CB     Centre-Bottom (halfway line, bottom edge)
      5    BL     Bottom-Left corner

The CT–CB segment represents the halfway line of the pitch.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def pick_calibration_frame(cap: "cv2.VideoCapture") -> Optional[np.ndarray]:
    """
    Step through video frames one at a time so the user can choose one suitable
    for calibration (avoiding logos, replays, close-ups).

    Controls:
        SPACE — use the current frame
        N     — advance one frame
        F     — fast-forward 30 frames
        ESC   — cancel

    Returns:
        The chosen frame, or None if cancelled / stream exhausted.
    """
    win = "Select calibration frame"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    ret, frame = cap.read()
    if not ret or frame is None:
        cv2.destroyWindow(win)
        return None

    while True:
        display = frame.copy()
        cv2.putText(
            display,
            "SPACE: use frame   N: next   F: +30   ESC: cancel",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow(win, display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord(" "):
            cv2.destroyWindow(win)
            return frame
        if key in (ord("n"), ord("N")):
            ret, new = cap.read()
            if ret and new is not None:
                frame = new
        elif key in (ord("f"), ord("F")):
            for _ in range(30):
                ret, new = cap.read()
                if ret and new is not None:
                    frame = new
                else:
                    break
        elif key == 27:
            cv2.destroyWindow(win)
            return None

_POINT_LABELS = ["TL", "CT", "TR", "BR", "CB", "BL"]


class FieldCalibrator:
    """
    Interactive drag-and-drop calibration tool for a 6-point pitch polygon.

    Points can be dragged freely — including outside the video frame — to
    accommodate any camera angle or zoom level.
    """

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame.copy()
        self.h, self.w = frame.shape[:2]

        self.padding = max(200, int(self.h * 0.3))
        self.canvas_w = self.w + 2 * self.padding
        self.canvas_h = self.h + 2 * self.padding
        self.offset_x = self.padding
        self.offset_y = self.padding

        margin_x = int(self.w * 0.1)
        margin_y = int(self.h * 0.15)
        cx = self.w // 2

        self.points = np.array(
            [
                [margin_x, margin_y],                     # 0: TL
                [cx, margin_y],                           # 1: CT
                [self.w - margin_x, margin_y],            # 2: TR
                [self.w - margin_x, self.h - margin_y],   # 3: BR
                [cx, self.h - margin_y],                  # 4: CB
                [margin_x, self.h - margin_y],            # 5: BL
            ],
            dtype=np.float32,
        )

        self.dragging: bool = False
        self.active_point: Optional[int] = None
        self.hover_point: Optional[int] = None
        self._WIN = "Field Calibration"

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    # Visual constants (BGR)
    _OVERLAY_BLUE = (220, 130, 50)       # translucent fill for inside-polygon area
    _LINE_WHITE = (240, 240, 240)        # polygon edges
    _HALFWAY_CYAN = (220, 220, 60)       # CT–CB halfway line
    _HANDLE_YELLOW = (0, 230, 255)       # idle handle
    _HANDLE_ORANGE = (0, 145, 255)       # hovered or dragged handle
    _HANDLE_RING = (255, 255, 255)       # handle outline
    _INSTR_BG = (24, 24, 28)             # instruction bar background

    def draw_frame(self) -> np.ndarray:
        """Return the current canvas with polygon, control handles and labels."""
        canvas = np.full((self.canvas_h, self.canvas_w, 3), (32, 32, 36), dtype=np.uint8)
        canvas[
            self.offset_y : self.offset_y + self.h,
            self.offset_x : self.offset_x + self.w,
        ] = self.frame.copy()

        pts = (self.points + np.array([self.offset_x, self.offset_y])).astype(np.int32)

        # Translucent blue polygon overlay + outer dimming
        overlay = canvas.copy()
        cv2.rectangle(
            overlay,
            (self.offset_x, self.offset_y),
            (self.offset_x + self.w, self.offset_y + self.h),
            (0, 0, 0),
            -1,
        )
        cv2.fillPoly(overlay, [pts], self._OVERLAY_BLUE)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

        cv2.rectangle(
            canvas,
            (self.offset_x, self.offset_y),
            (self.offset_x + self.w - 1, self.offset_y + self.h - 1),
            (200, 200, 200),
            2,
        )

        # White polygon edges
        for i in range(len(pts)):
            cv2.line(
                canvas,
                tuple(pts[i]),
                tuple(pts[(i + 1) % len(pts)]),
                self._LINE_WHITE,
                2,
                cv2.LINE_AA,
            )

        # Halfway line: CT (1) → CB (4)
        if len(pts) >= 5:
            cv2.line(canvas, tuple(pts[1]), tuple(pts[4]), self._HALFWAY_CYAN, 2, cv2.LINE_AA)

        # Handles — orange on hover or drag, yellow otherwise
        handle_size = 12
        for idx, (pt, label) in enumerate(zip(pts, _POINT_LABELS)):
            x, y = int(pt[0]), int(pt[1])
            is_active = idx == self.active_point and self.dragging
            is_hover = idx == self.hover_point and not self.dragging
            color = self._HANDLE_ORANGE if (is_active or is_hover) else self._HANDLE_YELLOW
            radius = handle_size + 2 if (is_active or is_hover) else handle_size
            cv2.circle(canvas, (x, y), radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), radius, self._HANDLE_RING, 2, cv2.LINE_AA)
            cv2.putText(canvas, str(idx), (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (x - 15, y - radius - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Bottom instruction bar
        bar_h = 36
        bar_top = self.canvas_h - bar_h
        cv2.rectangle(canvas, (0, bar_top), (self.canvas_w, self.canvas_h), self._INSTR_BG, -1)
        cv2.line(canvas, (0, bar_top), (self.canvas_w, bar_top), (90, 90, 100), 1)
        cv2.putText(
            canvas,
            "CLICK + DRAG points | SPACE: confirm | R: reset | ESC: cancel",
            (12, bar_top + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        return canvas

    # ------------------------------------------------------------------
    # Point manipulation
    # ------------------------------------------------------------------

    def get_point_index(self, x: int, y: int, threshold: int = 20) -> Optional[int]:
        """Return the index of the nearest control point within *threshold* pixels."""
        best_idx: Optional[int] = None
        best_dist = float(threshold)
        for idx, pt in enumerate(self.points):
            d = float(np.hypot(x - pt[0], y - pt[1]))
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx

    def update_point(self, point_idx: int, x: int, y: int) -> None:
        """Move control point *point_idx* to canvas-relative coordinates (x, y)."""
        self.points[point_idx] = [x, y]

    def _reset_points(self) -> None:
        margin_x = int(self.w * 0.1)
        margin_y = int(self.h * 0.15)
        cx = self.w // 2
        self.points = np.array(
            [
                [margin_x, margin_y],
                [cx, margin_y],
                [self.w - margin_x, margin_y],
                [self.w - margin_x, self.h - margin_y],
                [cx, self.h - margin_y],
                [margin_x, self.h - margin_y],
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Interactive calibration loop
    # ------------------------------------------------------------------

    def calibrate(self) -> np.ndarray:
        """
        Open the calibration window and block until the user confirms or cancels.

        Returns:
            Array of shape (6, 2) with the calibrated point coordinates.
        """
        print("\n" + "=" * 70)
        print("FIELD CALIBRATION - FUTSAL")
        print("=" * 70)
        print("\nCONTROL POINTS:")
        for i, label in enumerate(_POINT_LABELS):
            descriptions = {
                "TL": "Top-Left corner",
                "CT": "Centre-Top (halfway line)",
                "TR": "Top-Right corner",
                "BR": "Bottom-Right corner",
                "CB": "Centre-Bottom (halfway line)",
                "BL": "Bottom-Left corner",
            }
            print(f"  {i} = {label:2s}  —  {descriptions[label]}")
        print("\nCONTROLS:")
        print("  CLICK + DRAG  Move a point")
        print("  SPACE         Confirm calibration")
        print("  R             Reset to defaults")
        print("  ESC           Cancel")
        print("=" * 70 + "\n")

        cv2.namedWindow(self._WIN)

        def _mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
            frame_x = x - self.offset_x
            frame_y = y - self.offset_y

            if event == cv2.EVENT_LBUTTONDOWN:
                idx = self.get_point_index(frame_x, frame_y)
                if idx is not None:
                    self.dragging = True
                    self.active_point = idx
                    self.hover_point = idx
                    print(f"  -> Point {_POINT_LABELS[idx]} ({idx}) active")
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.dragging and self.active_point is not None:
                    self.update_point(self.active_point, frame_x, frame_y)
                else:
                    self.hover_point = self.get_point_index(frame_x, frame_y)
            elif event == cv2.EVENT_LBUTTONUP:
                if self.dragging and self.active_point is not None:
                    print(f"  [OK] Point {_POINT_LABELS[self.active_point]} ({self.active_point}) placed")
                self.dragging = False
                self.active_point = None
                self.hover_point = self.get_point_index(frame_x, frame_y)

        cv2.setMouseCallback(self._WIN, _mouse)
        print("Window open. Drag the yellow handles to the pitch boundaries.\n")

        while True:
            cv2.imshow(self._WIN, self.draw_frame())
            key = cv2.waitKey(50) & 0xFF

            if key == ord(" "):
                print("[OK] Calibration confirmed\n")
                break
            elif key in (ord("r"), ord("R")):
                self._reset_points()
                print("[OK] Points reset\n")
            elif key == 27:
                print("[CANCEL] Calibration cancelled\n")
                break

        cv2.destroyAllWindows()
        return self.points.copy()


def load_calibration(path: Path) -> np.ndarray:
    """Load a previously-saved 6-point calibration array from a .npy file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    points = np.load(path).astype(np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected calibration array of shape (N, 2), got {points.shape}")
    if len(points) < 4:
        raise ValueError(f"Calibration file has only {len(points)} points (need at least 4)")
    logger.info("Loaded %d calibration points from %s", len(points), path)
    return points


def save_calibration(points: np.ndarray, path: Path) -> None:
    """Save a calibration array to a .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, points.astype(np.float32))
    logger.info("Saved %d calibration points to %s", len(points), path)


# ---------------------------------------------------------------------------
# Entry point for `futsal-calibrate` console script
# ---------------------------------------------------------------------------

def run_standalone() -> None:
    """
    Standalone calibration tool — entry point for the ``futsal-calibrate`` command.

    Accepts a local video file path or a YouTube URL. Saves the resulting
    6-point array to ``calibration_points.npy`` in the current directory.
    """
    import sys

    print("\n" + "=" * 70)
    print(" " * 20 + "FUTSAL FIELD CALIBRATOR")
    print("=" * 70 + "\n")

    source = input("Enter video path or YouTube URL: ").strip()
    if not source:
        logger.error("Empty input — aborting")
        sys.exit(1)

    # Resolve YouTube URLs via yt-dlp
    if "youtube.com" in source or "youtu.be" in source:
        from futsal_analytics.config import Config
        from futsal_analytics.stream import open_youtube_stream

        cap = open_youtube_stream(source, Config())
    else:
        cap = cv2.VideoCapture(source)

    if cap is None or not cap.isOpened():
        logger.error("Could not open video source: %s", source)
        sys.exit(1)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        logger.error("Could not read the first frame")
        sys.exit(1)

    logger.info("Frame dimensions: %dx%d", frame.shape[1], frame.shape[0])

    calibrator = FieldCalibrator(frame)
    points = calibrator.calibrate()

    print("\n" + "=" * 70)
    print("CALIBRATED POINTS:")
    print("=" * 70)
    for i, (pt, label) in enumerate(zip(points, _POINT_LABELS)):
        print(f"  {i} ({label:2s}): x={pt[0]:7.2f}, y={pt[1]:7.2f}")

    output = Path("calibration_points.npy")
    save_calibration(points, output)
    print(f"\n[OK] Points saved to: {output}")
    print("[DONE] Calibration complete.\n")
