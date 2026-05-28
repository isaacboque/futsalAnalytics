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

The YOLO model (`yolo11n.pt`, ~5.6 MB) is downloaded automatically by
`ultralytics` on first run and cached locally. Override the model with
`Config(model_name="yolo11s.pt")` or similar for accuracy / speed trade-offs.

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
| `--snapshot-every N` | Write annotated camera + tactical board JPEGs every N frames for web live previews (0 = disabled) | `0` |
| `--snapshot-dir DIR` | Where to write live snapshots (defaults to the same parent as `--save-positions`) | inferred |
| `--snapshot-width PX` | Downscale snapshots to this width before encoding (0 = native) | `960` |
| `--imgsz PX` | YOLO inference input size (longer side). Lower = faster, may miss small players | `640` |
| `--frame-stride N` | Process every Nth frame (1 = every frame). Direct 1/N throughput multiplier | `1` |

The CLI auto-detects local video file paths as `--url`, so the same `futsal-analytics`
command works for both YouTube and local sources.

The CLI also accepts a **local video file path** as `--url` — local paths are
opened directly by OpenCV without going through yt-dlp.

## Web app

A three-page Streamlit app:

- **Analyse** — paste a YouTube URL or upload a local video, pick a clean
  frame, place 6 calibration points in the browser, and start the analyser.
  The CLI runs as a subprocess with `--no-gui` and stdout streams into a
  live log panel (with live camera + tactical-board previews).
- **Viewer** — interactive Overview, Tracks/Players, Heatmaps, Replay
  (with optional synced camera video), and Video tabs for a single run.
- **Roster** — define each team's roster once, assign track IDs to real
  players, and the Viewer's dashboards roll up automatically per-player.

```bash
pip install -e ".[viewer]"
streamlit run web/app.py --server.address 0.0.0.0 --server.port 8501
```

The sidebar nav switches between Home / Analyse / Viewer. See
[`web/README.md`](web/README.md) for the full workflow and file layout.

## Technical Details

### Core Algorithms & Formulas

#### **Perspective Transformation (6-Point Homography)**

A homography is a 3×3 projective transformation matrix **H** that maps camera-space coordinates to tactical board coordinates:

```
src = field_rect[:6]  # [TL, CT, TR, BR, CB, BL] in camera space
dst = np.array([
    [0, 0],                    # TL
    [width/2, 0],              # CT (halfway line)
    [width, 0],                # TR
    [width, height],           # BR
    [width/2, height],         # CB (halfway line)
    [0, height]                # BL
])
H, _ = cv2.findHomography(src, dst, method=0)  # least-squares
```

**Why 6 points instead of 4?**
- 4-corner perspective assumes straight lines remain straight (pure perspective).
- 6-point homography is more general and better captures camera distortion.
- Halfway-line points (CT, CB) constrain mid-pitch geometry → more accurate.

**Coordinate System:**
- **Camera space:** Video frame (0 → 1920 pixels typical)
- **Board space:** 700 × 350 pixels (40 m × 20 m futsal pitch at 2:1 aspect ratio)

#### **Point-in-Polygon Field Validation**

Uses ray casting + OpenCV `pointPolygonTest()` with dual validation:

1. **Fast axis-aligned bounding box pre-filter**
2. **Expensive point-in-polygon test (only if bbox passes)**

A player detection is kept if **either** foot position **or** bounding-box centroid is inside the polygon.

#### **Team Classification via K-Means Clustering**

Assigns players to teams based on jersey color in HSV space (more robust to lighting than RGB).

**Feature Extraction:**
1. Crop upper 40% of player bounding box (jersey region)
2. Convert BGR → HSV
3. Filter by saturation: Keep pixels with S > 30 (colored, not grayscale)
4. Compute mean HSV feature vector

**K-Means partitions into 3 clusters:**
- **Clusters 0 & 1:** Teams A and B (largest clusters)
- **Cluster 2 (smallest):** Referee

```python
from sklearn.cluster import KMeans
features = np.array([get_jersey_hsv(frame, bbox) for bbox in detections])
kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
kmeans.fit(features)
labels = kmeans.predict(features)
ref_label = np.argmin(np.bincount(labels))  # referee = smallest cluster
```

#### **PlayerTracker: Hungarian Assignment (Greedy Variant)**

Maintains persistent player IDs across frames using:

1. **Cost Matrix:** Euclidean distances between predicted track positions and new detections
2. **Velocity Prediction:** Extrapolates position using previous velocity
3. **Greedy Hungarian:** Sorts pairs by cost, assigns if both unassigned, rejects if cost > 150 pixels
4. **Smoothing:** Exponential moving average (α = 0.6) on positions
5. **Track Lifecycle:** Creation on unmatched detection, persistence for 10 frames, deletion if unmatched for 10+ frames

#### **BallTracker: Kalman Filter**

Smooths ball position and predicts during occlusion using 1D Kalman filter per axis.

**State:** [x, y, vx, vy] (position + velocity in 2D board space)

**Prediction step:**
```
x_pred = x_{t-1} + v_{t-1} * Δt
p_pred = p_{t-1} + q  (q = 0.01 process noise)
```

**Correction step (if measurement available):**
```
y = z - x_pred           (residual)
s = p_pred + r           (innovation covariance, r = 4.0 measurement noise)
k = p_pred / s           (Kalman gain)
x_corrected = x_pred + k * y
p_corrected = (1 - k) * p_pred
```

**No-Detection Handling:** Predicts up to 5 frames without detection; resets if discontinuity > 150 pixels.

#### **KPI Calculations (in metres)**

**Pixel-to-Metre Conversion:**
```
m_per_px_x = 40 m / 700 px ≈ 0.0571 m/px
m_per_px_y = 20 m / 350 px ≈ 0.0571 m/px
```

**Distance Covered:**
```
d_total = Σ ||p_t - p_{t-1}||_2
```
Anomaly filtering: Skip if step distance > 10 m (impossible sprint).

**Top Speed:**
```
v_max = max_t (||p_t - p_{t-1}||_2 / Δt)
```
where Δt = 1/fps (frame duration in seconds).

**Sprint Count:**
```
sprint_count = # transitions from v < 5.0 m/s to v ≥ 5.0 m/s
```
Default threshold: 5.0 m/s (typical futsal sprint).

**Possession:**
```
possession_time = Σ 1[closest_to_ball AND d < 1.5m] / fps
```

**Duel Time:**
```
duel_time = Σ 1[within 1.5m of opposing_team_player] / fps
```

### Pitch Specifications

- **Physical dimensions:** 40 m × 20 m (FIFA futsal standard)
- **Board rendering:** 700 × 350 pixels (2:1 aspect ratio)
- **Markings:** D-shaped penalty areas (6m quarter-circles), 3m centre circle, 6m/10m marks, halfway line

### Module Map

| Module | Purpose | Algorithm |
|--------|---------|-----------|
| `stream.py` | YouTube stream extraction | yt-dlp wrapper + OpenCV |
| `calibration.py` | Interactive 6-point calibration | Tkinter UI |
| `field.py` | Field validation & mapping | Point-in-polygon + homography |
| `detection.py` | YOLO inference, team classification | YOLOv11 + K-Means |
| `board.py` | Pitch rendering | OpenCV drawing + FIFA spec |
| `kpis.py` | Per-player metrics | Euclidean distance + statistics |
| `config.py` | Central configuration | Dataclass |

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
├── .streamlit/config.toml
├── src/futsal_analytics/
│   ├── __init__.py
│   ├── __main__.py        argparse CLI + main pipeline + run-lock + snapshots
│   ├── config.py          Config dataclass
│   ├── stream.py          stream / local-file opening + HH:MM:SS parsing
│   ├── calibration.py     FieldCalibrator (OpenCV UI) + load/save helpers
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
│   ├── test_field.py            (incl. 6-point homography numerics)
│   ├── test_kpis.py
│   ├── test_runtime_helpers.py  (snapshot writer + lock file)
│   ├── test_stream.py           (parse_start_time, local-file fast path)
│   └── test_web_shared.py       (robust JSONL / CSV loaders)
├── web/
│   ├── app.py             landing page
│   ├── _shared.py         CSS, theme, cached loaders, sidebar helpers
│   ├── README.md
│   └── pages/
│       ├── 1_Analyse.py   URL/local upload → in-browser calibration → run
│       ├── 2_Viewer.py    Overview / Tracks(/Players) / Heatmaps / Replay / Video
│       └── 3_Roster.py    Track ID → real player assignment
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
