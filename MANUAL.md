# Futsal Analytics — Complete User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Installation & Setup](#installation--setup)
3. [Quick Start](#quick-start)
4. [CLI Reference](#cli-reference)
5. [Web App Guide](#web-app-guide)
6. [Calibration Guide](#calibration-guide)
7. [Understanding Outputs](#understanding-outputs)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)
10. [Technical Architecture](#technical-architecture)

---

## Introduction

**Futsal Analytics** is a real-time match analysis system that processes futsal (indoor soccer) videos from YouTube or local files. It provides:

- **Player Detection** using YOLO11 neural network
- **Persistent Tracking** using ByteTrack algorithm
- **Team Classification** via K-Means clustering on jersey colors (HSV)
- **Perspective Mapping** with 6-point homography for camera-to-board transformation
- **Performance Metrics** including distance covered, speed, sprints, possession time, and duel time
- **Outputs**: Annotated videos, position data (JSONL), and per-player statistics (CSV)

### Key Features

| Feature | Description |
|---------|-------------|
| **YOLO11 Detection** | Detects players and ball from any camera angle |
| **6-Point Calibration** | Interactive calibration for any side-camera angle |
| **Persistent IDs** | Same player keeps same track ID across frames |
| **Team Auto-Classification** | HSV-based clustering with adaptive retraining |
| **Tactical Board** | FIFA-spec futsal pitch rendering (D-shaped penalties, markings) |
| **Per-Player KPIs** | Distance, speed, sprints, possession, duel time |
| **GPU Acceleration** | Automatic GPU detection; CPU fallback available |
| **Headless Mode** | Batch processing without GUI for CI/automation |

---

## Installation & Setup

### Prerequisites

- **Python**: 3.9 or higher
- **Git**: For cloning the repository
- **Disk Space**: ~2 GB (including YOLO models and dependencies)
- **GPU** (optional): NVIDIA CUDA for faster processing

### Step 1: Clone the Repository

```bash
git clone https://github.com/isaacboque/futsalAnalytics.git
cd futsalAnalytics
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install the Package

**For CLI use only:**
```bash
pip install -e .
```

**For Web App (recommended):**
```bash
pip install -e ".[viewer]"
```

**For Development (with tests):**
```bash
pip install -e ".[dev]"
```

### Step 4: Verify Installation

```bash
futsal-analytics --help
futsal-calibrate --help
```

Both commands should display help text.

### Dependencies

- `opencv-python` — Video processing
- `ultralytics` — YOLO detection
- `supervision` — Object tracking utilities
- `scikit-learn` — K-Means clustering
- `yt-dlp` — YouTube video streaming
- `numpy` — Numerical computing
- `streamlit` — Web app (optional)
- `plotly` — Interactive charts (optional)
- `pandas` — Data analysis (optional)

---

## Quick Start

### Option 1: Interactive Mode (Easiest)

```bash
futsal-analytics
```

You will be prompted for:
1. **YouTube URL** or local video path
2. **Start time** (MM:SS format, optional)

Then:
1. The **calibration UI** opens with 6 yellow handles
2. **Drag the handles** to match the pitch corners and halfway line
3. **Press SPACE** to confirm and start analysis
4. **Watch the progress** — processing happens in real-time

Results are saved to the current directory.

### Option 2: Web App (Recommended for UI)

```bash
streamlit run web/app.py
```

Then:
1. Open your browser to the displayed URL (usually `http://localhost:8501`)
2. Go to **Analyse** from the sidebar
3. Enter your YouTube URL
4. Pick a clean frame and place calibration points in the browser
5. Click **Start analysis**

The web app provides:
- Visual calibration with click-to-place
- Live progress logs
- Real-time preview videos
- Post-analysis viewer with interactive dashboards

### Option 3: Headless/Batch Mode (Scripted)

Fastest for batch processing or CI/CD:

```bash
# Step 1: Calibrate once (interactive)
futsal-calibrate

# Step 2: Run headless against any video with saved calibration
futsal-analytics \
    --url https://www.youtube.com/watch?v=VIDEO_ID \
    --calibration calibration_points.npy \
    --no-gui \
    --device cuda \
    --save-video output/camera.mp4 \
    --save-board-video output/board.mp4 \
    --save-positions output/positions.jsonl \
    --save-kpis output/kpis.csv
```

---

## CLI Reference

### Basic Invocation

```bash
futsal-analytics [OPTIONS]
```

### Input Options

| Flag | Type | Description | Default |
|------|------|-------------|---------|
| `--url` | STRING | YouTube URL or local video path | Prompts if omitted |
| `--start` | TIME | Start time in MM:SS or seconds | `0` |

### Calibration Options

| Flag | Type | Description | Default |
|------|------|-------------|---------|
| `--calibration PATH` | FILE | Load pre-saved `.npy` calibration file | — |
| `--save-calibration PATH` | FILE | Save the interactive calibration result | — |
| `--allow-frame-pick` | BOOL | Step through frames before calibrating (skip logos/replays) | off |

### Output Options

| Flag | Type | Description | Default |
|------|------|-------------|---------|
| `--save-video PATH** | FILE | Annotated camera video (MP4) | — |
| `--save-board-video PATH` | FILE | Tactical board video (MP4) | — |
| `--save-positions PATH` | FILE | Per-frame positions as JSONL | — |
| `--save-kpis PATH` | FILE | Per-player statistics as CSV | — |
| `--snapshot-every N` | INT | Write JPEGs every N frames (0=disabled) | `0` |
| `--snapshot-dir DIR` | PATH | Directory for live preview JPEGs | Auto-inferred |
| `--snapshot-width PX` | INT | Downscale snapshots to this width | `960` |

### Processing Options

| Flag | Type | Description | Default |
|------|------|-------------|---------|
| `--no-gui` | BOOL | Headless mode (requires `--calibration`) | off |
| `--device` | {auto,cpu,cuda} | Compute device for YOLO | `auto` |
| `--imgsz PX` | INT | YOLO inference size (longer side). Lower=faster | `640` |
| `--max-frames N` | INT | Stop after N frames (0=unlimited) | `0` |
| `--frame-stride N` | INT | Process every Nth frame (1=every frame) | `1` |
| `--retrain-every N` | INT | Retrain team classifier every N frames (0=never) | `0` |
| `--skip-when-behind SEC` | FLOAT | For live streams, skip ahead when lagging | `0.0` |

### Logging

| Flag | Type | Description | Default |
|------|------|-------------|---------|
| `--log-level** | {DEBUG,INFO,WARNING,ERROR} | Console verbosity | `INFO` |

### Examples

**Example 1: Quick YouTube analysis with defaults**
```bash
futsal-analytics --url https://www.youtube.com/watch?v=ABC123
```

**Example 2: Start at 15:30, use GPU, limit to 1000 frames**
```bash
futsal-analytics \
    --url https://www.youtube.com/watch?v=ABC123 \
    --start 15:30 \
    --device cuda \
    --max-frames 1000 \
    --save-video out/camera.mp4 \
    --save-kpis out/kpis.csv
```

**Example 3: Process local file with saved calibration**
```bash
futsal-analytics \
    --url /path/to/futsal_match.mp4 \
    --calibration ./calibration_points.npy \
    --no-gui \
    --save-positions out/positions.jsonl
```

**Example 4: Batch process multiple videos**
```bash
for video in videos/*.mp4; do
    futsal-analytics \
        --url "$video" \
        --calibration cal.npy \
        --no-gui \
        --save-kpis "out/$(basename "$video" .mp4)_kpis.csv"
done
```

---

## Web App Guide

### Installation

```bash
pip install -e ".[viewer]"
```

### Running

```bash
streamlit run web/app.py
```

Opens browser at `http://localhost:8501`

### Navigation

- **Sidebar**: Switch between **Home** / **Analyse** / **Viewer**

### Analyse Page

Complete workflow from video source to live analysis:

#### Step 1: Source Selection

1. Choose source type:
   - **YouTube URL**: Paste link to YouTube futsal match
   - **Upload File**: Upload local MP4 video
2. **Start time** (optional): MM:SS or seconds (e.g., "15:30" or "930")
3. **Output directory**: Where results and calibration will be saved

#### Step 2: Pick Calibration Frame

1. Click **Open stream** to load the video
2. Navigate to a wide-angle shot showing the entire pitch:
   - **Next frame** — Go to next frame
   - **Skip 30 s** — Jump ahead 30 seconds
   - **Skip 60 s** — Jump ahead 60 seconds
3. Avoid frames with:
   - Logos or graphics overlaid
   - Replay mode
   - Zoomed-in or tactical camera
4. Click **Frame looks good** to confirm

#### Step 3: Place Calibration Points

The 6-point calibration maps camera view to tactical board:

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

**Instructions:**
1. **Adjust margin slider** (if needed) to place points outside the frame
2. **Click in order**: TL → CT → TR → BR → CB → BL
3. **Fine-tune points** by clicking in the right-hand list to re-place individually
4. **Save calibration** to save as `cal.npy`

Tips:
- Place points at **exact pitch boundaries** for accuracy
- CT and CB represent the **halfway line**
- If camera crops a corner, place point **outside frame** using margin slider

#### Step 4: Run Analysis

1. **Choose outputs**:
   - ☑ Save camera video (annotated with player boxes)
   - ☑ Save board video (tactical board view)
   - ☑ Save position data (JSONL per frame)
   - ☑ Save KPI statistics (CSV per player)

2. **Compute options**:
   - **Device**: auto / cpu / cuda
   - **Retrain classifier every N frames**: 0 (never) or (e.g., 500)
   - **Max frames**: 0 (unlimited) or (e.g., 3000)

3. **Start analysis**
   - Live log shows progress
   - Camera + board previews update every 30 frames
   - Click **Stop** to terminate cleanly

### Viewer Page

Post-analysis interactive dashboards for a single run:

#### Setup

1. **Output directory**: Point to folder containing `positions.jsonl`
2. **Track filter**: Hide short ID-switch fragments (minimum track length)
3. **Reload data** if you've updated the output directory

#### Tabs

| Tab | Purpose |
|-----|---------|
| **Overview** | Summary stats (duration, frame count, avg players detected) |
| **Tracks/Players** | Timeline of all track IDs with detection count and team |
| **Heatmaps** | 2D density plots of player movement by team |
| **Replay** | Frame-by-frame replay with optional synced camera video |
| **Video** | Play camera and/or board videos inline |

#### Navigation

- **Scrub timeline** at the bottom of most tabs
- **Hover** over elements for details
- **Filter by team** or specific track ID
- **Download** raw CSV data from Tracks/Players tab

---

## Calibration Guide

Calibration is the **critical one-time setup** per camera angle. It maps camera coordinates to a standard overhead pitch layout.

### Understanding the Six Points

The 6-point calibration uses:
- **4 pitch corners** (TL, TR, BR, BL)
- **2 halfway-line points** (CT, CB) for mid-pitch accuracy

Why 6 instead of 4?
- 4-point perspective assumes straight lines stay straight
- 6-point homography adapts to camera lens distortion
- Halfway-line points constrain mid-pitch geometry

### Web Calibration (Recommended)

**Pros:**
- Visual feedback with click-to-place
- Re-place individual points without redoing the whole calibration
- Adjustable margin to place points outside the visible frame

**Process:**

```bash
streamlit run web/app.py
```

1. Go to **Analyse** page
2. Enter YouTube URL or upload video
3. Click **Open stream** → navigate to good frame
4. Adjust **Click margin (%)** slider if camera crops corners
5. Click 6 times: `TL → CT → TR → BR → CB → BL`
6. Click **Save calibration** → saves to `cal.npy`

### OpenCV Calibration (Legacy Desktop)

For minimal dependencies:

```bash
futsal-calibrate
# or
python scripts/calibrate.py /path/to/video.mp4
```

**Controls:**

| Key | Action |
|-----|--------|
| **Click + drag** | Move point |
| **SPACE** | Confirm calibration |
| **R** | Reset to defaults |
| **ESC** | Cancel |

### Reusing Calibrations

Once calibrated, reuse the same `.npy` file for all videos from that camera:

```bash
futsal-analytics \
    --url video_2.mp4 \
    --calibration calibration_points.npy \
    --no-gui
```

### Tips for Accurate Calibration

1. **Use a wide-angle frame** — Show the entire pitch
2. **Avoid reflections** — No wet areas or glare
3. **Place corners precisely** — At the exact pitch boundary
4. **Halfway-line accuracy** — CT and CB should be at the center
5. **Test with a short run** — Use `--max-frames 100` first
6. **Visual validation** — Check if players move realistically on the board video

### Troubleshooting Calibration

**Players drift off the field:**
- Recalibrate with more precision at corners
- Check that halfway-line points (CT, CB) are exactly centered

**Ball appears in wrong location:**
- Fine-tune the corner points
- Retrain the team classifier with `--retrain-every 500`

**Perspective looks warped:**
- This is normal for non-parallel side-camera angles
- Adjust top vs. bottom points slightly

---

## Understanding Outputs

### 1. Annotated Camera Video (`camera.mp4`)

Raw video with overlays:
- **Bounding boxes** around detected players (color by team)
- **Track ID** and **team label** above each box
- **Speed** displayed (m/s)
- **Ball position** as small circle

Use to verify detection and tracking quality visually.

### 2. Tactical Board Video (`board.mp4`)

Overhead view of detected positions mapped to a standard pitch:
- **Pitch markings** (D-shaped penalty areas, 6m/10m lines, center circle)
- **Player positions** as colored dots (size = speed)
- **Team clusters** in distinct colors
- **Ball** shown as small white circle
- **Frame number** in corner

Use to analyze team formations and movement patterns.

### 3. Position Data (`positions.jsonl`)

Per-frame positions in **JSON Lines** format (one JSON object per line):

```json
{
  "frame": 0,
  "timestamp": 0.0,
  "detections": [
    {
      "track_id": 1,
      "team": 0,
      "bbox": [100, 50, 150, 200],
      "center_camera": [125, 125],
      "center_board": [20, 10],
      "speed": 0.5,
      "class": "person"
    },
    ...
  ],
  "ball": {
    "center_camera": [300, 150],
    "center_board": [60, 7],
    "detected": true
  }
}
```

**Fields:**
- `frame` — Frame number (0-indexed)
- `timestamp` — Seconds from video start
- `track_id` — Persistent player ID across frames
- `team` — 0 or 1 (auto-classified by jersey color)
- `bbox` — [x, y, width, height] in camera space
- `center_camera` — [x, y] in pixel space
- `center_board` — [x, y] in board space (meters scaled)
- `speed` — Current frame-to-frame speed (m/s)

**Usage:**
- Load with pandas or JSON parser for custom analysis
- Plot player trajectories
- Statistical analysis by team/player
- Export to other sports analytics tools

### 4. Per-Player KPIs (`kpis.csv`)

**Columns:**

| Column | Description | Unit |
|--------|-------------|------|
| `track_id` | Persistent player identifier | — |
| `team` | Team classification (0 or 1) | — |
| `total_distance` | Distance player covered | meters (m) |
| `top_speed` | Highest frame-to-frame speed | m/s |
| `sprint_count` | Number of frames exceeding 5 m/s | count |
| `possession_time` | Approximate time player held ball proximity | seconds (s) |
| `duel_time` | Time in close proximity to opponent | seconds (s) |

**Example:**
```csv
track_id,team,total_distance,top_speed,sprint_count,possession_time,duel_time
1,0,450.2,7.1,12,15.3,45.8
2,0,380.5,6.8,8,12.1,52.3
3,1,520.1,8.2,18,9.5,38.7
...
```

**Usage:**
- Compare players' fitness levels (distance, top_speed)
- Identify high-intensity periods (sprint_count)
- Possession distribution across team
- Pressure/defense intensity (duel_time)

---

## Advanced Features

### GPU Acceleration

Enable faster processing on NVIDIA GPUs:

```bash
futsal-analytics --device cuda
```

**Performance impact:**
- **CPU (default)**: ~1-3 fps for 1080p video
- **GPU (CUDA)**: ~10-30 fps for 1080p video (8-10x speedup)
- **Mobile devices**: CPU only (no CUDA support)

The `--device auto` flag (default) auto-detects your GPU.

### Frame Skipping (`--frame-stride`)

Process every Nth frame to speed up analysis:

```bash
futsal-analytics --frame-stride 3  # Process every 3rd frame
```

**Tradeoff:**
- 3x faster processing
- 3x fewer output frames
- Slightly lower tracking continuity

Use for quick previews or long matches where real-time isn't critical.

### Model Selection

Override the default YOLO11n model:

```bash
futsal-analytics --imgsz 640  # Larger input = better accuracy, slower
futsal-analytics --imgsz 320  # Smaller input = faster, may miss small players
```

Internally, you can also modify `Config(model_name="yolo11s.pt")` for speed/accuracy trade-offs.

### Team Classifier Retraining

By default, team classification is done once at the start. For long matches or lighting changes, retrain periodically:

```bash
futsal-analytics --retrain-every 500  # Retrain every 500 frames
```

This adapts to gradual lighting changes (sun movement, stadium lights, etc.).

### Live Streaming Mode

For live YouTube streams, skip ahead when processing lags:

```bash
futsal-analytics \
    --url https://www.youtube.com/watch?v=LIVE_STREAM_ID \
    --skip-when-behind 5.0  # Skip ahead if >5s behind
```

Useful for real-time broadcasts where you don't care about dropped frames.

### Batch Processing

Process multiple videos with a loop:

```bash
for url in $(cat urls.txt); do
    futsal-analytics \
        --url "$url" \
        --calibration cal.npy \
        --no-gui \
        --device cuda \
        --save-kpis "out/$(date +%s)_kpis.csv" \
        --log-level WARNING
done
```

### Integration with Custom Code

The core modules can be imported for custom pipelines:

```python
from futsal_analytics.stream import open_youtube_stream
from futsal_analytics.calibration import load_calibration
from futsal_analytics.detection import process_frame, setup_detectors
from futsal_analytics.kpis import KPITracker

# Load video and calibration
cap = open_youtube_stream(url)
calib = load_calibration("calibration_points.npy")

# Setup detection pipeline
detector, tracker, classifier, ball_tracker = setup_detectors()

# Process frames
for frame_idx in range(max_frames):
    ret, frame = cap.read()
    if not ret:
        break
    
    detections = process_frame(frame, detector, tracker, classifier)
    # ... custom processing
```

---

## Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError: No module named 'futsal_analytics'"

**Cause:** Package not installed or venv not activated

**Fix:**
```bash
# Activate venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install
pip install -e .
```

#### 2. "YouTube URL not accessible" or "yt-dlp error"

**Cause:** Video blocked, private, or restricted; yt-dlp outdated

**Fix:**
```bash
# Update yt-dlp
pip install --upgrade yt-dlp

# Try with local file instead
futsal-analytics --url /path/to/video.mp4

# Check YouTube URL is valid and public
```

#### 3. Out-of-memory error

**Cause:** Processing very long video or GPU memory exhausted

**Fix:**
```bash
# Use CPU instead of GPU
futsal-analytics --device cpu

# Limit frames
futsal-analytics --max-frames 1000

# Reduce inference size
futsal-analytics --imgsz 320

# Use frame skipping
futsal-analytics --frame-stride 2
```

#### 4. Players detected incorrectly (wrong boxes, missed players)

**Cause:** Lighting, resolution, unusual angles

**Fix:**
- Use a larger YOLO model: modify `Config(model_name="yolo11s.pt")`
- Check calibration is accurate (see [Calibration Guide](#calibration-guide))
- Increase `--imgsz` for better accuracy (slower)
- Retrain team classifier: `--retrain-every 500`

#### 5. Calibration points off or perspective warped

**Cause:** Imprecise point placement or extreme camera angle

**Fix:**
- Place points at **exact pitch corners** (not inside)
- Ensure CT and CB are at **exact halfway line**
- For extreme angles, place points **outside the frame** (use margin slider)
- Re-calibrate with more care

#### 6. Ball not detected or jumps around

**Cause:** Ball too small, fast motion, or occlusion

**Fix:**
- This is a known limitation of frame-by-frame detection
- Temporal smoothing is applied internally
- Not critical for team/player KPIs
- Consider ignoring ball data if unreliable

#### 7. Streamlit web app slow or freezing

**Cause:** Large videos or insufficient hardware

**Fix:**
```bash
# Reduce snapshot resolution
# (in web/app.py, adjust snapshot_width parameter)

# Use frame skipping for faster preview
# (set frame_stride in web UI)

# Process on GPU
# (select "cuda" in web UI device dropdown)
```

#### 8. "CUDA out of memory"

**Cause:** GPU memory exhausted

**Fix:**
```bash
# Fall back to CPU
futsal-analytics --device cpu

# Reduce batch size (YOLO internal)
# Reduce inference size
futsal-analytics --imgsz 384
```

### Debug Logging

Enable verbose logging:

```bash
futsal-analytics --log-level DEBUG 2>&1 | tee analysis.log
```

This writes frame-by-frame diagnostics to `analysis.log` for inspection.

### Performance Profiling

For slow processing, check the bottleneck:

```bash
time futsal-analytics --url video.mp4 --max-frames 100
```

Compare wall-time vs. frame count to estimate FPS.

---

## Technical Architecture

### Processing Pipeline

```
YouTube/Local Video
        │
        ▼
Stream Reader (yt-dlp / OpenCV)
        │
        ▼
Calibration (6-point homography)
        │
        ├─ FieldValidator (polygon filter)
        └─ SimpleFieldMapper (camera→board transform)
        │
        ▼
Per-Frame Detection Loop
        ├─ YOLO11 Inference
        ├─ Size + Polygon Filtering
        ├─ ByteTrack Assignment
        ├─ HSV Team Classifier (K-Means)
        ├─ Ball Temporal Smoothing
        ├─ KPI Accumulation
        └─ Tactical Board Rendering
        │
        ▼
Outputs (Video, JSONL, CSV)
```

### Key Algorithms

#### Perspective Transformation (6-Point Homography)

Maps camera coordinates to a standard 700×350 px board (40m × 20m pitch):

```python
src = np.array([TL, CT, TR, BR, CB, BL])  # Camera space
dst = np.array([
    [0, 0],              # TL
    [350, 0],            # CT (halfway)
    [700, 0],            # TR
    [700, 350],          # BR
    [350, 350],          # CB (halfway)
    [0, 350]             # BL
])
H, _ = cv2.findHomography(src, dst)  # 3×3 matrix

# Apply to point p_camera
p_board = cv2.perspectiveTransform(p_camera, H)
```

**Why 6 points?**
- 4-point perspective is limited to pure geometric projection
- 6-point homography adapts to lens distortion
- Halfway-line constraints ensure accurate mid-pitch geometry

#### Point-in-Polygon Field Validation

Only keeps detections inside the pitch polygon:

```python
# Fast bounding-box pre-filter
if point not in bbox:
    continue

# Expensive point-in-polygon test (only if bbox passes)
if cv2.pointPolygonTest(polygon, point) >= 0:
    keep_detection()
```

A detection is valid if **either foot or centroid** is inside the polygon.

#### Team Classification (K-Means on HSV)

Clusters players into 2 teams based on jersey color:

```python
# Extract HSV values from detection bounding boxes
hsv_values = [...]  # Shape: (N, 3)

# Fit K-Means with k=2 clusters
kmeans = KMeans(n_clusters=2)
kmeans.fit(hsv_values)

# Assign team labels
team_labels = kmeans.labels_
```

**Why HSV instead of RGB?**
- HSV separates hue (color) from saturation/brightness (lighting)
- RGB is sensitive to shadows and reflections
- Adaptive retraining handles lighting changes

#### ByteTrack Persistent IDs

Assigns continuous track IDs across frames:

```python
# Frame N: detections [d1, d2, d3]
# Frame N+1: detections [d1', d2']

# ByteTrack computes Hungarian assignment
# Tracks high-confidence matches: d1→d1', d2→d2'
# Creates new track if no match or removes if disappears
```

**Advantages:**
- Robust to occlusions (temporary disappearance)
- Handles identity switches gracefully
- No manual re-annotation needed

### File Formats

#### Calibration File (`.npy`)

NumPy binary array of shape `(6, 2)`:
```python
points = np.array([
    [x_TL, y_TL],
    [x_CT, y_CT],
    [x_TR, y_TR],
    [x_BR, y_BR],
    [x_CB, y_CB],
    [x_BL, y_BL]
], dtype=np.float32)

np.save("calibration.npy", points)
```

Load with:
```python
points = np.load("calibration.npy")
```

#### Positions JSONL (`.jsonl`)

One JSON object per line:
- Each line is a valid JSON object
- Lines are **not** separated by commas
- Total file size can be large (10s of MB)

Parse with:
```python
import json
with open("positions.jsonl") as f:
    for line in f:
        frame_data = json.loads(line)
        print(frame_data)
```

#### KPIs CSV (`.csv`)

Standard CSV with header row. Columns:
- `track_id` — INT
- `team` — INT (0 or 1)
- `total_distance` — FLOAT (meters)
- `top_speed` — FLOAT (m/s)
- `sprint_count` — INT (frames > 5 m/s)
- `possession_time` — FLOAT (seconds)
- `duel_time` — FLOAT (seconds)

---

## Tips & Best Practices

### Video Selection

1. **Choose wide-angle shots** — Calibration works best when you see the entire pitch
2. **Avoid replays or tactical overlays** — These confuse detection
3. **Use HD or 4K** — 1080p or higher for small player detection
4. **Steady camera** — Avoid pans or zooms during calibration frame

### Calibration Quality

1. **Place points at exact boundaries** — Not inside the pitch
2. **Halfway-line points must be centered** — CT and CB are critical
3. **Test with short run first** — `--max-frames 100` to verify
4. **Reuse same calibration** — Save it for future videos from same camera

### Processing Optimization

1. **Use GPU when available** — 8-10x speedup
2. **Batch process at night** — GPU load lower, faster turnaround
3. **Enable frame skipping for quick previews** — `--frame-stride 2`
4. **Use smaller model for quick tests** — `yolo11n.pt` (default) is good balance

### Data Analysis

1. **Filter outliers** — Some track IDs are very short (ID switches)
2. **Normalize by frame count** — Different runs may have different durations
3. **Team-level aggregation** — Sum distance/sprints per team for team metrics
4. **Possession approximation** — Use proximity to ball (custom analysis)

### Troubleshooting Workflow

1. Start with **short video** (1-2 min) to iterate fast
2. Use **calibration frame picker** to ensure clean frame
3. **Save camera video** (`--save-video`) to visually inspect detection
4. **Save board video** (`--save-board-video`) to verify calibration accuracy
5. Once working, scale to full matches with `--max-frames 0`

---

## Command Cheat Sheet

```bash
# Interactive mode (prompts for URL)
futsal-analytics

# Interactive with URL directly
futsal-analytics --url https://www.youtube.com/watch?v=ABC

# Calibrate once
futsal-calibrate

# Headless with all outputs
futsal-analytics --url video.mp4 \
    --calibration cal.npy \
    --no-gui \
    --device cuda \
    --save-video out/camera.mp4 \
    --save-board-video out/board.mp4 \
    --save-positions out/pos.jsonl \
    --save-kpis out/kpis.csv

# Quick preview (100 frames only)
futsal-analytics --url video.mp4 \
    --calibration cal.npy \
    --no-gui \
    --max-frames 100

# Fast processing (every 3rd frame)
futsal-analytics --url video.mp4 \
    --calibration cal.npy \
    --no-gui \
    --frame-stride 3

# Web app
streamlit run web/app.py

# Debug logging
futsal-analytics --log-level DEBUG --url video.mp4 --calibration cal.npy --no-gui

# Batch process directory
for f in videos/*.mp4; do
    futsal-analytics --url "$f" --calibration cal.npy --no-gui \
        --save-kpis "out/$(basename "$f" .mp4)_kpis.csv"
done
```

---

## Support & Resources

- **GitHub Issues**: https://github.com/isaacboque/futsalAnalytics/issues
- **Documentation**: See [docs/](docs/) folder
- **Sample Videos**: See [docs/SAMPLE_VIDEOS.md](docs/SAMPLE_VIDEOS.md)
- **Changelog**: See [docs/CHANGELOG.md](docs/CHANGELOG.md)

---

**Last Updated**: May 2026  
**Version**: 0.1.0  
**Author**: Isaac Boque
