"""Field geometry: polygon validation and 6-point homography mapping."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FieldValidator:
    """
    Determines whether detected players are inside the calibrated pitch polygon.

    Accepts the 6-point array produced by ``FieldCalibrator.calibrate()``:
    ``[TL, CT, TR, BR, CB, BL]``.
    """

    def __init__(self, field_polygon: np.ndarray) -> None:
        if len(field_polygon) < 4:
            raise ValueError(f"At least 4 calibration points required, got {len(field_polygon)}")
        if len(field_polygon) < 6:
            logger.warning("Expected 6 calibration points, got %d", len(field_polygon))

        self.field_polygon = field_polygon.astype(np.float32)
        self.x_min = float(np.min(self.field_polygon[:, 0]))
        self.x_max = float(np.max(self.field_polygon[:, 0]))
        self.y_min = float(np.min(self.field_polygon[:, 1]))
        self.y_max = float(np.max(self.field_polygon[:, 1]))

        logger.info(
            "FieldValidator: %d-point polygon, bbox [%.0f,%.0f]→[%.0f,%.0f]",
            len(self.field_polygon),
            self.x_min,
            self.y_min,
            self.x_max,
            self.y_max,
        )

    def is_within_field(self, point: np.ndarray) -> bool:
        """Return True if *point* [x, y] lies on or inside the pitch polygon."""
        if point[0] < self.x_min or point[0] > self.x_max:
            return False
        if point[1] < self.y_min or point[1] > self.y_max:
            return False
        return cv2.pointPolygonTest(self.field_polygon, (float(point[0]), float(point[1])), False) >= 0

    def filter_detections(self, detections, feet_coords: np.ndarray) -> np.ndarray:
        """
        Build a boolean mask of detections whose player is on the pitch.

        A detection is kept when either the foot position **or** the bounding-box
        centre lies within the pitch polygon.
        """
        valid = []
        for i, foot in enumerate(feet_coords):
            foot_ok = self.is_within_field(foot)
            bbox = detections.xyxy[i]
            centre = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            centre_ok = self.is_within_field(centre)
            valid.append(foot_ok or centre_ok)
            if not (foot_ok or centre_ok):
                logger.debug("Filtered out detection %d: foot=%s, centre=%s", i, foot_ok, centre_ok)
        return np.array(valid)


class SimpleFieldMapper:
    """
    Maps camera-space coordinates to tactical-board coordinates.

    Uses **all 6 calibration points** (TL, CT, TR, BR, CB, BL) as
    correspondences into a least-squares homography via ``cv2.findHomography``.
    This is more accurate than the 4-corner ``getPerspectiveTransform`` because
    the halfway-line points (CT, CB) constrain the mid-pitch geometry.

    Destination layout on the board:

        TL=(0, 0)              CT=(W/2, 0)              TR=(W, 0)
        BL=(0, H)              CB=(W/2, H)              BR=(W, H)
    """

    def __init__(self, field_rect: np.ndarray, board_width: int, board_height: int) -> None:
        """
        Args:
            field_rect: Array of shape (>=4, 2). Six points use the 6-point
                        homography; four points fall back to the corner-only
                        perspective transform.
            board_width: Width of the output tactical board in pixels.
            board_height: Height of the output tactical board in pixels.
        """
        self.board_width = board_width
        self.board_height = board_height

        if len(field_rect) >= 6:
            src = field_rect[:6].astype(np.float32)
            dst = np.array(
                [
                    [0, 0],                                 # TL
                    [board_width / 2.0, 0],                 # CT
                    [board_width, 0],                       # TR
                    [board_width, board_height],            # BR
                    [board_width / 2.0, board_height],      # CB
                    [0, board_height],                      # BL
                ],
                dtype=np.float32,
            )
            self.M, _ = cv2.findHomography(src, dst, method=0)
            self._mode = "6-point homography (least-squares)"
        elif len(field_rect) >= 4:
            corners = field_rect[:4].astype(np.float32)
            dst = np.array(
                [[0, 0], [board_width, 0], [board_width, board_height], [0, board_height]],
                dtype=np.float32,
            )
            self.M = cv2.getPerspectiveTransform(corners, dst)
            self._mode = "4-point perspective"
        else:
            raise ValueError(f"At least 4 calibration points required, got {len(field_rect)}")

        if self.M is None:
            raise RuntimeError("Failed to compute homography (degenerate point configuration?)")

        logger.info("SimpleFieldMapper: %s, %d input points", self._mode, len(field_rect))

    def transform(self, pt: np.ndarray) -> np.ndarray:
        """Map a single camera-space point to board coordinates."""
        if len(pt) < 2 or pt[0] < 0 or pt[1] < 0:
            logger.debug("Invalid point: %s", pt)
            return np.array([0.0, 0.0])
        try:
            result = cv2.perspectiveTransform(pt.reshape(1, 1, 2).astype(np.float32), self.M)[0][0]
            if result[0] < 0 or result[0] > self.board_width or result[1] < 0 or result[1] > self.board_height:
                logger.debug("Transformed point outside board bounds: %s", result)
            return result
        except Exception as exc:
            logger.error("Transform error: %s", exc)
            return np.array([0.0, 0.0])

    def transform_batch(self, pts: np.ndarray) -> np.ndarray:
        """Map an array of shape (N, 2) of camera points to board coordinates."""
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        reshaped = pts.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.M)
        return transformed.reshape(-1, 2)
