# Field Calibration

Futsal Analytics maps a side-camera frame onto an overhead 40 m × 20 m
tactical board via a **6-point homography**. You define those 6 points once
per camera angle and the result is saved as a `.npy` file you can re-use for
every future analysis from the same broadcast.

You can calibrate in two ways:

1. **In the browser** — recommended, via the **Analyse** page of the Streamlit
   app. Click-to-place + per-point re-place + an adjustable click margin so
   points can sit outside the visible frame.
2. **OpenCV window** — legacy, via the `futsal-calibrate` console script or
   automatically when you run `futsal-analytics` without `--calibration`.

Either path produces the same `(6, 2)` array of pixel coordinates.

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

The segment **CT–CB** represents the halfway line; including it gives the
homography solver enough information to constrain mid-pitch geometry, which
a 4-corner perspective transform cannot.

---

## Web calibration (recommended)

```bash
pip install -e ".[viewer]"
streamlit run web/app.py
```

Then in the browser:

1. Open the **Analyse** page from the sidebar.
2. Source: paste a YouTube URL or upload a local MP4.
3. Click **Open stream**, then **Next frame** / **Skip 30 s** / **Skip 60 s**
   to land on a wide-angle shot that shows the whole pitch.
4. Optionally widen the **Click margin around the frame (%)** slider so you
   can place points outside the visible frame (useful when the camera crops
   a pitch corner — broadcast standard).
5. Click 6 times in order: `TL → CT → TR → BR → CB → BL`. Each placed point
   appears in the right-hand list. Click any list entry to **re-place** that
   single point without redoing the others.
6. Click **Save calibration**. The points are written to
   `<output-dir>/cal.npy`.

You can then start the run on the same page (it will pick up the freshly
saved `cal.npy` automatically) or invoke the CLI with `--calibration
<output-dir>/cal.npy`.

---

## OpenCV calibration

If you don't want to install the `[viewer]` extras, the legacy desktop UI
still works:

```bash
# After pip install -e .
futsal-calibrate

# Or directly without an install
python scripts/calibrate.py
python scripts/calibrate.py /path/to/video.mp4
python scripts/calibrate.py https://www.youtube.com/watch?v=VIDEO_ID
```

### Controls

| Key / Action          | Effect                                |
|-----------------------|---------------------------------------|
| Click + drag a point  | Move the control handle freely        |
| SPACE                 | Confirm calibration and continue      |
| R                     | Reset all points to default positions |
| ESC                   | Cancel and exit                       |

Points can be dragged **outside the visible frame** to handle wide-angle
cameras or when pitch corners are partially off-screen.

The saved 6-point array goes to `calibration_points.npy` in the current
directory.

```python
import numpy as np
points = np.load("calibration_points.npy")
# points.shape == (6, 2)
# points[0] == [x_TL, y_TL], etc.
```

---

## Tips for a good calibration

- **Fixed side-camera view**. The tool assumes a single fixed angle, the
  standard futsal broadcast layout. Multi-camera coverage is not supported.
- **Pick a wide-angle frame** without close-ups or replays. Use the frame
  picker (`--allow-frame-pick` for the CLI, the **Next / Skip 30 s / Skip
  60 s** buttons in the browser) to skip past intro graphics.
- **Halfway-line accuracy matters**. CT and CB constrain which half of the
  pitch each player maps to. A few pixels of slop here = several metres of
  error at the far touchline.
- **Wider polygon = fewer false rejections**. The `FieldValidator` drops
  detections outside the polygon, so if players near the sidelines are being
  filtered out, extend `TL` / `BL` / `TR` / `BR` outward (use the click
  margin to place them beyond the visible frame).
- **Re-use across matches**. The same camera angle gives the same
  `cal.npy`; copy it into each match's output directory to skip
  re-calibration.
