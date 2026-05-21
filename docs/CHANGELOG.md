# Changelog

## Unreleased

### Repository restructure

- Moved all application code into `src/futsal_analytics/` package.
- Split `futsal_analyzer.py` into focused modules:
  `config`, `stream`, `calibration`, `field`, `detection`, `board`, `__main__`.
- Replaced duplicated `FieldCalibrator` in `calibracion_simple.py` with a
  shared import from `futsal_analytics.calibration`.
- Added `pyproject.toml` with declared dependencies and console-script entry points.
- Added `pytest`-based test suite under `tests/`.
- Translated all documentation to English; moved docs to `docs/`.

---

## Previous development history

### Calibration: 4-point → 6-point polygon

- **Problem**: Original calibrator used only 4 corner points, providing no
  information about the halfway line.
- **Change**: Added two additional control points — CT (Centre-Top) and CB
  (Centre-Bottom) — which mark the intersection of the halfway line with the
  top and bottom pitch edges.
- **Point order** (current): `TL, CT, TR, BR, CB, BL`.

### Mouse interaction fix

- **Problem**: Mouse click events were being dropped because the drag state was
  stored in local variables inside the callback closure, which were not shared
  correctly across events.
- **Fix**: Moved `dragging` and `active_point` to instance attributes
  (`self.dragging`, `self.active_point`), ensuring the callback always reads the
  correct shared state.

### FieldValidator: polygon-based validation

- Switched from a simple bounding-box check to `cv2.pointPolygonTest` against
  the full 6-point polygon.
- A detection is kept when **either** the foot position or the bounding-box
  centre lies within the polygon, reducing false rejections near sidelines.

### Lazy ML imports

- `ultralytics` (YOLO) and `scikit-learn` (KMeans) are now imported only when
  `setup_detectors` or `TeamClassifier.__init__` is first called, so the
  calibration UI opens quickly without loading the full ML stack.

### Removed features (not implemented in current codebase)

The following features were planned but are **not present** in the current code:

- ByteTrack / persistent player IDs across frames.
- Distance, duel, possession, and sprint KPI tracking.
- CSV export of match statistics.
