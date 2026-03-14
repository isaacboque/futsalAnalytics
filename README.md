# Football Match Analysis System

Advanced video analysis system for football (soccer) matches using AI-powered computer vision. Extracts tactical positioning, player performance metrics, and real-time match statistics from YouTube streams.

## Features

- **YOLO11 Detection**: Real-time player and ball detection using YOLOv11n
- **Multi-Object Tracking**: ByteTrack for persistent player identification across frames
- **Team Classification**: Automatic team assignment using HSV color-based K-Means clustering
- **Perspective Transformation**: Piecewise homography for accurate 2D tactical board mapping
- **KPI Computation**: 
  - Distance covered (in metres)
  - Duel frequency (player-to-player proximity events)
  - Possession time (seconds)
  - Sprint detection (>7 m/s threshold)
- **Live Visualization**: 
  - Camera view with bounding boxes
  - Tactical board with player positions
- **CSV Export**: Exportable match statistics

## Installation

### Requirements
- Python 3.8+
- NVIDIA GPU (optional, for faster inference)
- 2GB+ RAM

### Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install opencv-python numpy supervision ultralytics scikit-learn yt-dlp

# Download YOLO model (optional - auto-downloads on first run)
pip install ultralytics
yolo detect download model=yolov11n.pt
```

## Usage

```bash
python "# LATEST.py"
```

### Interactive Workflow

1. **Input YouTube URL**: Provide a public football match video URL
2. **Optional Start Time**: Skip to a specific time (MM:SS format)
3. **Field Calibration**: Click 6 points on the first frame:
   - Top-left corner
   - Top-right corner
   - Bottom-right corner
   - Bottom-left corner
   - Halfway line (top)
   - Halfway line (bottom)
4. **Live Monitoring**: Watch real-time detection and tactical mapping
5. **Export Results**: Press Q to stop and save KPI statistics to CSV

## Configuration

Edit the `Config` dataclass in the script to customize:

```python
config = Config(
    board_width=800,           # Tactical board dimensions
    board_height=400,
    smoothing_window=5,        # Position smoothing (frames)
    duel_radius_px=40,         # Proximity threshold for duels
    possession_radius_px=50,   # Proximity threshold for possession
    yolo_conf_threshold=0.3,   # Detection confidence
)
```

## Architecture

- **TeamClassifier**: K-Means clustering on HSV jersey colors (identifies 2 teams + referee)
- **PositionSmoother**: Temporal averaging to reduce detection jitter
- **PiecewiseTransformer**: Split domain perspective transformation for accuracy
- **KPITracker**: Accumulates performance metrics per player
- **TacticalBoard**: Real-time rendering of match state

## Output

**CSV Export Format** (`match_data.csv`):
- `track_id`: Player identifier
- `team`: Team assignment (0, 1, or -1 for referee)
- `distance_m`: Total distance covered (metres)
- `duel_frames`: Number of dueling events
- `possession_s`: Total possession time (seconds)
- `sprint_count`: Number of sprint events

## Troubleshooting

### "yt-dlp not found"
```bash
pip install --upgrade yt-dlp
```

### "Could not open stream"
- Verify the URL is public (not age-restricted)
- Update yt-dlp: `pip install --upgrade yt-dlp`

### Poor detection performance
- Ensure adequate lighting in the video
- Adjust `yolo_conf_threshold` in Config (lower = more detections, more false positives)
- Use higher quality video resolution

## Technical Notes

- **Pitch Scale**: 800px board = 40m real width (1px = 0.05m)
- **Sprint Threshold**: ~7 m/s (calculated frame-delta dependent)
- **Duel Detection**: Pairwise distance comparison at 40px radius
- **Referee Identification**: Smallest K-Means cluster (typically different jersey color)

## License

MIT

## Author

Isaac Boque - isboque19@gmail.com
