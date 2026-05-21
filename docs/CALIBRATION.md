# Field Calibration

Futsal Analytics uses a **6-point interactive calibration** to define the exact
pitch boundary in the camera frame. This allows the system to:

- Filter out detections that are outside the playing area.
- Map camera-space player positions to the overhead tactical board.

---

## The Six Control Points

```
        1 (CT)
       /      \
      /        \
    0 (TL)    2 (TR)
    |          |
    |          |
    5 (BL)    3 (BR)
      \        /
       \      /
        4 (CB)
```

| Index | Label | Description                              |
|-------|-------|------------------------------------------|
| 0     | TL    | Top-Left corner of the pitch             |
| 1     | CT    | Centre-Top — halfway line, top edge      |
| 2     | TR    | Top-Right corner of the pitch            |
| 3     | BR    | Bottom-Right corner of the pitch         |
| 4     | CB    | Centre-Bottom — halfway line, bottom edge|
| 5     | BL    | Bottom-Left corner of the pitch          |

The segment CT–CB represents the halfway line.

---

## Controls

| Key / Action          | Effect                                |
|-----------------------|---------------------------------------|
| Click + drag a point  | Move the control handle freely        |
| SPACE                 | Confirm calibration and continue      |
| R                     | Reset all points to default positions |
| ESC                   | Cancel and exit                       |

Points can be dragged **outside the visible frame** to handle wide-angle cameras
or when pitch corners are partially off-screen.

---

## Running the calibrator standalone

```bash
# After pip install -e .
futsal-calibrate

# Or directly as a script (no install required)
python scripts/calibrate.py
python scripts/calibrate.py /path/to/video.mp4
python scripts/calibrate.py https://www.youtube.com/watch?v=VIDEO_ID
```

The calibrated points are saved to `calibration_points.npy` in the current
directory.

```python
import numpy as np
points = np.load("calibration_points.npy")
# points.shape == (6, 2)
# points[0] == [x_TL, y_TL], etc.
```

---

## Tips

- **Fixed side camera**: The tool is designed for a single fixed side-camera
  view, the standard futsal broadcast angle.
- **Accuracy matters for the halfway line**: Position CT and CB precisely on
  the halfway line — these determine which half of the board each player maps to.
- **Wider polygon = fewer false rejections**: If players near the sidelines are
  being filtered out, extend TL/BL outward.
