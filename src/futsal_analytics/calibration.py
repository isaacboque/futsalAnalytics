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
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

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
        self._WIN = "Field Calibration"

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_frame(self) -> np.ndarray:
        """Return the current canvas with polygon, control handles and labels."""
        canvas = np.full((self.canvas_h, self.canvas_w, 3), (80, 80, 80), dtype=np.uint8)
        canvas[
            self.offset_y : self.offset_y + self.h,
            self.offset_x : self.offset_x + self.w,
        ] = self.frame.copy()

        pts = (self.points + np.array([self.offset_x, self.offset_y])).astype(np.int32)

        # Darken area outside the polygon
        overlay = canvas.copy()
        cv2.rectangle(
            overlay,
            (self.offset_x, self.offset_y),
            (self.offset_x + self.w, self.offset_y + self.h),
            (0, 0, 0),
            -1,
        )
        cv2.fillPoly(overlay, [pts], (50, 100, 50))
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

        cv2.rectangle(
            canvas,
            (self.offset_x, self.offset_y),
            (self.offset_x + self.w - 1, self.offset_y + self.h - 1),
            (200, 200, 200),
            2,
        )

        for i in range(len(pts)):
            cv2.line(canvas, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 255, 0), 3)

        # Halfway line: CT (1) → CB (4)
        if len(pts) >= 5:
            cv2.line(canvas, tuple(pts[1]), tuple(pts[4]), (0, 200, 200), 2, cv2.LINE_AA)

        handle_size = 12
        for idx, (pt, label) in enumerate(zip(pts, _POINT_LABELS)):
            x, y = int(pt[0]), int(pt[1])
            color = (0, 165, 255) if (self.active_point == idx and self.dragging) else (0, 255, 255)
            cv2.circle(canvas, (x, y), handle_size, color, -1)
            cv2.circle(canvas, (x, y), handle_size, (255, 255, 255), 3)
            cv2.putText(canvas, str(idx), (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(canvas, label, (x - 15, y - handle_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(
            canvas,
            "CLICK + DRAG points | SPACE: confirm | R: reset | ESC: cancel",
            (self.offset_x + 10, self.offset_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas

    # ------------------------------------------------------------------
    # Point manipulation
    # ------------------------------------------------------------------

    def get_point_index(self, x: int, y: int, threshold: int = 25) -> Optional[int]:
        """Return the index of the nearest control point within *threshold* pixels."""
        for idx, pt in enumerate(self.points):
            if np.sqrt((x - pt[0]) ** 2 + (y - pt[1]) ** 2) < threshold:
                return idx
        return None

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
                    print(f"  -> Point {_POINT_LABELS[idx]} ({idx}) active")
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.dragging and self.active_point is not None:
                    self.update_point(self.active_point, frame_x, frame_y)
            elif event == cv2.EVENT_LBUTTONUP:
                if self.dragging and self.active_point is not None:
                    print(f"  [OK] Point {_POINT_LABELS[self.active_point]} ({self.active_point}) placed")
                self.dragging = False
                self.active_point = None

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

    output = "calibration_points.npy"
    np.save(output, points)
    print(f"\n[OK] Points saved to: {output}")
    print("[DONE] Calibration complete.\n")
