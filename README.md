# Futsal Analytics

[![tests](https://github.com/isaacboque/futsalAnalytics/actions/workflows/tests.yml/badge.svg)](https://github.com/isaacboque/futsalAnalytics/actions/workflows/tests.yml)

Real-time futsal match analysis from YouTube streams.
YOLO detection + ByteTrack persistent IDs + HSV team clustering +
6-point perspective mapping + per-player KPIs.

## Features

- **YOLO11 detection** of players and ball from a YouTube stream
- **Interactive 6-point field calibration** — drag handles to fit any side-camera angle
- **6-point homography** (`cv2.findHomography`) — uses halfway-line points as additional
  constraints for a more accurate camera → board mapping than a 4-corner perspective
- **ByteTrack persistent IDs** — same player keeps the same `track_id` across frames
- **Team classification** via K-Means on HSV jersey colours; optional periodic re-training
  to adapt to lighting changes
- **FIFA-spec futsal pitch** rendering on the tactical board (D-shaped penalty areas,
  6 m / 10 m marks, 3 m centre circle)
- **Per-player KPIs**: distance covered (m), top speed (m/s), sprint count,
  possession (s), duel time (s)
- **Outputs**: annotated MP4 video, per-frame positions JSONL, per-player KPIs CSV
- **Headless mode** for batch / CI usage
- **GPU acceleration** via `--device cuda` (auto-detected by default)

## Pipeline

```
YouTube URL
    │
    ▼
open_youtube_stream()              stream.py
    │
    ▼
calibration (UI or .npy file)      calibration.py
    │
    ├── FieldValidator              field.py        polygon filter
    └── SimpleFieldMapper           field.py        6-point homography
    │
    ▼
process_frame()                    detection.py
    ├── YOLO predict
    ├── size + polygon filter
    ├── ByteTrack assign IDs
    ├── TeamClassifier              detection.py    K-Means on HSV
    ├── BallTracker                 detection.py    temporal smoothing
    ├── KPITracker.update()         kpis.py
    └── TacticalBoard.draw_state()  board.py
```

## Installation

```bash
git clone https://github.com/isaacboque/futsalAnalytics.git
cd futsalAnalytics

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -e .
# add the test extras if you plan to contribute
pip install -e ".[dev]"
```

Dependencies (declared in `pyproject.toml`): `opencv-python`, `ultralytics`,
`supervision`, `scikit-learn`, `yt-dlp`, `numpy`.

The YOLO model (`yolo11n.pt`) is downloaded automatically on first run.

## Usage

### Quick start (interactive)

```bash
futsal-analytics
# or equivalently
python -m futsal_analytics
```

You'll be prompted for a URL, then the calibration UI opens; drag the
6 yellow handles to the pitch boundary, press SPACE, and analysis begins.

### Fully scripted (no prompts, no GUI)

```bash
# 1. Calibrate once and save the result
futsal-calibrate                     # produces calibration_points.npy

# 2. Run headless against any video, reusing that calibration
futsal-analytics \
    --url https://www.youtube.com/watch?v=VIDEO_ID \
    --calibration calibration_points.npy \
    --no-gui \
    --device cuda \
    --max-frames 3000 \
    --save-video out/camera.mp4 \
    --save-board-video out/board.mp4 \
    --save-positions out/positions.jsonl \
    --save-kpis out/kpis.csv
```

### All CLI flags

| Flag | Description | Default |
|------|-------------|---------|
| `--url` | YouTube URL (prompts if omitted) | — |
| `--start` | Start time `MM:SS` or seconds | `0` |
| `--calibration PATH` | Load saved 6-point `.npy`, skip the UI | — |
| `--save-calibration PATH` | Save the result of interactive calibration | — |
| `--allow-frame-pick` | Step through frames before calibrating (skip logos/replays) | off |
| `--save-video PATH` | Write annotated camera video (MP4) | — |
| `--save-board-video PATH` | Write tactical board video (MP4) | — |
| `--save-positions PATH` | Per-frame positions JSONL | — |
| `--save-kpis PATH` | Per-player KPIs CSV | — |
| `--no-gui` | Headless mode (requires `--calibration`) | off |
| `--device {auto,cpu,cuda}` | YOLO compute device | `auto` |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Verbosity | `INFO` |
| `--max-frames N` | Stop after N frames (0 = unlimited) | `0` |
| `--retrain-every N` | Re-train team classifier every N frames | `0` (never) |
| `--skip-when-behind SECONDS` | For live streams, skip ahead when processing lags | `0.0` (off) |

## Web viewer

A Streamlit dashboard loads the analyser's output files and shows interactive
overview metrics, per-player KPI tables, heatmaps, a frame-by-frame
tactical-board replay, and the recorded videos.

```bash
pip install -e ".[viewer]"
streamlit run web/app.py
```

In the sidebar, set **Output directory** to the folder containing
`positions.jsonl`, `kpis.csv`, and optional `board.mp4` / `camera.mp4`.
See [`web/README.md`](web/README.md) for the full workflow.

## Output formats

### KPIs CSV (`--save-kpis`)

| Column | Description |
|--------|-------------|
| `track_id` | Persistent ByteTrack ID |
| `team` | Team assignment (0, 1, or -1 for referee) |
| `distance_m` | Total distance covered, in metres (capped per frame to filter ID-swap jumps) |
| `top_speed_ms` | Peak per-frame speed, m/s |
| `sprint_count` | Number of sprint events (≥ 5 m/s) |
| `possession_s` | Seconds spent as nearest player to ball within 1.5 m |
| `duel_s` | Seconds spent within 1.5 m of an opposing-team player |
| `seen_s` | Seconds the track was visible |

### Positions JSONL (`--save-positions`)

One JSON object per video frame:

```json
{"frame": 0, "t": 0.0, "players": [{"id": 1, "team": 0, "x": 350.2, "y": 175.1}], "ball": {"x": 200.0, "y": 150.0}}
```

Coordinates are in tactical-board pixels (default `700 × 350` representing
a 40 m × 20 m pitch).

## Configuration

Override defaults via the `Config` dataclass:

```python
from futsal_analytics import Config

cfg = Config(
    board_width=1000,
    board_height=500,
    yolo_conf_threshold=0.25,    # lower = more detections
    min_players_for_kmeans=6,
)
```

You can pass this to `run()` programmatically; CLI flags take precedence
for I/O paths.

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Tests cover `FieldCalibrator` (load/save), `FieldValidator`, the new
6-point `SimpleFieldMapper`, `TeamClassifier`, `BallTracker`,
`TacticalBoard`, `KPITracker`, `PositionLogger`, and the argparse CLI —
without opening any GUI windows.

CI runs `pytest` on Python 3.10 / 3.11 / 3.12 against every push and PR.

## Project structure

```
futsalAnalytics/
├── pyproject.toml
├── README.md
├── .github/workflows/tests.yml
├── src/futsal_analytics/
│   ├── __init__.py
│   ├── __main__.py        argparse CLI + main pipeline
│   ├── config.py          Config dataclass
│   ├── stream.py          YouTube stream + frame reading
│   ├── calibration.py     FieldCalibrator + load/save helpers
│   ├── field.py           FieldValidator + SimpleFieldMapper (6-pt homography)
│   ├── detection.py       TeamClassifier, BallTracker, ByteTrack, process_frame
│   ├── board.py           TacticalBoard (FIFA-spec futsal markings)
│   └── kpis.py            KPITracker + PositionLogger
├── scripts/
│   └── calibrate.py       thin wrapper for `futsal-calibrate`
├── tests/
│   ├── conftest.py
│   ├── test_calibration.py
│   ├── test_cli.py
│   ├── test_detection.py
│   ├── test_field.py
│   └── test_kpis.py
└── docs/
    ├── CALIBRATION.md
    ├── CHANGELOG.md
    └── SAMPLE_VIDEOS.md
```

## Troubleshooting

**`yt-dlp not found`**
```bash
pip install --upgrade yt-dlp
```

**Stream fails to open**
- Verify the URL is public and not age-restricted.
- YouTube changes its API frequently — `pip install --upgrade yt-dlp` regularly.

**KPIs look wrong (huge distances, impossible top speed)**
- The board-pixel ↔ metres scale assumes a 40 m × 20 m pitch. If your pitch is
  significantly different, adjust `PITCH_WIDTH_M` / `PITCH_HEIGHT_M` in `board.py`.
- Sudden ID swaps create jumps. The per-frame speed cap (12 m/s) filters
  the most egregious ones; for stricter filtering, lower it in `kpis.py`.

**Live stream falls behind**
- Use `--skip-when-behind 2.0` to drop frames when lag exceeds 2 s.
- Use `--device cuda` if you have a GPU.

**Calibration is wrong because the first frame is a logo / replay**
- Use `--allow-frame-pick` and press `N` to advance frames until the pitch is
  clearly visible, then `SPACE`.

## License

MIT — see `pyproject.toml`.
