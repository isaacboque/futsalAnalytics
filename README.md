# Futsal Analytics

Real-time futsal match analysis from YouTube streams using YOLO object detection,
HSV-based team classification, and perspective-corrected tactical board rendering.

## Features

- **YOLO11 detection** — real-time player and ball detection from a YouTube stream
- **Interactive 6-point field calibration** — adapt to any fixed side-camera angle
- **Team colour classification** — K-Means on HSV jersey colours (2 teams + referee)
- **Perspective mapping** — projects camera-space positions to an overhead tactical board
- **Ball tracking** — temporal smoothing to stabilise ball position across frames
- **Dual live view** — camera view with bounding boxes + overhead tactical board

## Architecture

```
YouTube URL
    │
    ▼
open_youtube_stream()          stream.py
    │
    ▼
FieldCalibrator.calibrate()    calibration.py   (interactive UI — 6 drag points)
    │
    ├── FieldValidator          field.py         (polygon filter)
    └── SimpleFieldMapper       field.py         (homography → board coords)
    │
    ▼
process_frame() loop           detection.py
    ├── YOLO predict
    ├── TeamClassifier          detection.py     (K-Means on HSV)
    ├── BallTracker             detection.py     (temporal smoothing)
    └── TacticalBoard.draw_state()  board.py
```

## Installation

```bash
git clone https://github.com/your-user/futsalAnalytics.git
cd futsalAnalytics

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -e .
```

Dependencies are declared in `pyproject.toml`:
`opencv-python`, `ultralytics`, `supervision`, `scikit-learn`, `yt-dlp`, `numpy`.

The YOLO model (`yolo11n.pt`) is downloaded automatically on first run.

## Usage

### Full analysis pipeline

```bash
python -m futsal_analytics
# or, after pip install -e .
futsal-analytics
```

Interactive steps:
1. Enter a public YouTube URL.
2. Optionally enter a start time (`MM:SS`).
3. A calibration window opens — drag the 6 yellow handles to the pitch boundary.
4. Press **SPACE** to confirm. Analysis begins.
5. Press **Q** to stop.

### Calibration only

```bash
futsal-calibrate              # after pip install -e .
python scripts/calibrate.py  # or directly
```

Saves the 6-point array to `calibration_points.npy`.

## Configuration

Edit `Config` in `src/futsal_analytics/config.py` to adjust thresholds:

```python
from futsal_analytics import Config

cfg = Config(
    board_width=700,
    board_height=350,
    yolo_conf_threshold=0.3,   # lower = more detections
    min_players_for_kmeans=6,  # minimum players needed to train team classifier
)
```

Pass a custom `Config` to `main(cfg)` or use it as the default by editing the dataclass.

## Running the tests

```bash
pip install pytest
pytest tests/
```

Tests cover `FieldCalibrator`, `FieldValidator`, `SimpleFieldMapper`,
`BallTracker`, `TeamClassifier`, and `TacticalBoard` without opening any GUI
windows or requiring a GPU.

## Project structure

```
futsalAnalytics/
├── pyproject.toml
├── README.md
├── src/
│   └── futsal_analytics/
│       ├── __init__.py
│       ├── config.py       Config dataclass
│       ├── stream.py       YouTube stream opening, frame reading
│       ├── calibration.py  FieldCalibrator + run_standalone
│       ├── field.py        FieldValidator, SimpleFieldMapper
│       ├── detection.py    TeamClassifier, BallTracker, process_frame
│       ├── board.py        TacticalBoard
│       └── __main__.py     main() entry point
├── scripts/
│   └── calibrate.py        thin wrapper → futsal-calibrate entry point
├── tests/
│   ├── conftest.py
│   ├── test_calibration.py
│   ├── test_field.py
│   └── test_detection.py
└── docs/
    ├── CALIBRATION.md      calibration guide
    ├── CHANGELOG.md        version history
    └── SAMPLE_VIDEOS.md    test video links
```

## Troubleshooting

**`yt-dlp not found`**
```bash
pip install --upgrade yt-dlp
```

**Stream fails to open**
- Verify the URL is public and not age-restricted.
- Try `pip install --upgrade yt-dlp` — YouTube changes their API frequently.

**Poor detection**
- Lower `yolo_conf_threshold` to detect more players (more false positives too).
- Ensure the calibration polygon tightly encloses the visible pitch area.
- Higher-resolution video (720p+) improves detection accuracy.

**Team classifier not activating**
- Wait until at least `min_players_for_kmeans` (default 6) players are visible.
- If jersey colours are very similar, the classifier may refuse to train
  (insufficient colour variance).

## License

MIT — see `pyproject.toml` for author information.
