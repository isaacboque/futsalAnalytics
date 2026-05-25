# **Futsal Analytics - Complete Technical Documentation**

## **Executive Summary**

The Futsal Analytics system is an end-to-end computer vision pipeline for real-time futsal match analysis. It combines YOLOv11 object detection, 6-point perspective homography, multi-object tracking (Hungarian algorithm), K-Means team classification, and FIFA-standard pitch rendering to transform raw YouTube streams into tactical dashboards with per-player KPIs.

**Key Capabilities:**
- **Detection:** YOLOv11n (nano model) for players and ball
- **Tracking:** PlayerTracker (greedy Hungarian) + BallTracker (Kalman filter)
- **Team Classification:** K-Means clustering on HSV jersey colors
- **Perspective Mapping:** 6-point homography (more accurate than 4-corner)
- **KPIs:** Distance, speed, sprints, possession, duel time per player
- **Output:** MP4 videos, JSONL positions, CSV KPIs, interactive Streamlit dashboards

---

## **1. SYSTEM ARCHITECTURE**

### **1.1 Complete Pipeline**

```
YouTube Stream (yt-dlp)
    ↓
Frame Extraction (OpenCV VideoCapture)
    ↓
[Interactive Calibration] ← 6-point field polygon
    ↓
┌─────────────────────────────────────────────────────────┐
│           PER-FRAME PROCESSING LOOP                     │
├─────────────────────────────────────────────────────────┤
│ 1. YOLO Detection (ultralytics)                         │
│    └─ Detect players (class 0) & ball (class 32)        │
│                                                          │
│ 2. Detection Filtering                                  │
│    ├─ Size/aspect ratio validation                      │
│    └─ Field boundary check (FieldValidator)             │
│                                                          │
│ 3. Perspective Transformation                           │
│    └─ Camera space → Tactical board (SimpleFieldMapper) │
│                                                          │
│ 4. Tracking Assignment                                  │
│    ├─ PlayerTracker: Hungarian algorithm                │
│    └─ BallTracker: Kalman smoothing                      │
│                                                          │
│ 5. Team Classification                                  │
│    └─ TeamClassifier: K-Means on HSV                    │
│                                                          │
│ 6. KPI Accumulation                                     │
│    └─ Distance, speed, sprints, possession, duels       │
│                                                          │
│ 7. Visualization                                        │
│    ├─ Camera view (annotated bboxes + track IDs)        │
│    └─ Tactical board (2D overhead view)                 │
└─────────────────────────────────────────────────────────┘
    ↓
Output: MP4, JSONL, CSV
```

### **1.2 Module Overview**

| Module | Purpose | Algorithm |
|--------|---------|-----------|
| `stream.py` | YouTube stream extraction | yt-dlp wrapper + OpenCV |
| `calibration.py` | Interactive 6-point calibration | Mouse UI on Tkinter canvas |
| `field.py` | Field validation & mapping | Point-in-polygon + homography |
| `detection.py` | YOLO inference, team classification | YOLOv11 + K-Means clustering |
| `tracking.py` | Player tracking | Hungarian assignment |
| `ball.py` | Ball smoothing | Kalman filter (1D per axis) |
| `kpis.py` | Per-player metrics | Euclidean distance + statistics |
| `board.py` | Pitch rendering | OpenCV drawing + FIFA spec |
| `config.py` | Central configuration | Dataclass |

---

## **2. CORE ALGORITHMS & FORMULAS**

### **2.1 Perspective Transformation (6-Point Homography)**

**Problem:** Convert camera-space coordinates to tactical board coordinates.

**Solution:** Least-squares homography using all 6 calibration points (not just 4 corners).

**Mathematical Foundation:**

A homography is a 3×3 projective transformation matrix $\mathbf{H}$ that maps points from one plane to another:

$$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = \mathbf{H} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

Normalized output:
$$x_{\text{board}} = \frac{x'}{w'}, \quad y_{\text{board}} = \frac{y'}{w'}$$

**Implementation:**
```python
src = field_rect[:6]  # [TL, CT, TR, BR, CB, BL] in camera space
dst = np.array([
    [0, 0],                    # TL
    [width/2, 0],              # CT (halfway line)
    [width, 0],                # TR
    [width, height],           # BR
    [width/2, height],         # CB (halfway line)
    [0, height]                # BL
])
H, _ = cv2.findHomography(src, dst, method=0)  # method=0 = least-squares
```

**Why 6 points instead of 4?**
- 4-corner perspective assumes straight lines remain straight (pure perspective).
- 6-point homography is more general and better captures camera distortion.
- Halfway-line points (CT, CB) constrain mid-pitch geometry → more accurate.

**Coordinate System:**
- **Camera space:** Video frame (0 → 1920 pixels typical)
- **Board space:** 700 × 350 pixels (40 m × 20 m futsal pitch at 2:1 aspect ratio)

---

### **2.2 Point-in-Polygon Field Validation**

**Problem:** Filter detections to only those on the pitch.

**Algorithm:** Ray casting + OpenCV `pointPolygonTest()`

**Test:** For point $P = (x, y)$ and polygon $\mathcal{P}$:

$$\text{inside} = \begin{cases}
\text{True} & \text{if } P \text{ is inside or on boundary of } \mathcal{P} \\
\text{False} & \text{otherwise}
\end{cases}$$

**Implementation (optimized two-stage):**

1. **Fast axis-aligned bounding box pre-filter:**
   $$x_{\min} \leq x \leq x_{\max} \text{ AND } y_{\min} \leq y \leq y_{\max}$$

2. **Expensive point-in-polygon test (only if bbox passes):**
   ```python
   result = cv2.pointPolygonTest(polygon, (x, y), measureDist=False)
   is_inside = result >= 0
   ```

**Dual validation:** A player detection is kept if **either** foot position **or** bounding-box centroid is inside the polygon.

---

### **2.3 Team Classification via K-Means Clustering**

**Problem:** Assign players to two teams based on jersey color.

**Feature Space:** HSV (Hue, Saturation, Value) — more robust to lighting than RGB.

**Feature Extraction:**
1. Crop upper 40% of player bounding box (jersey region).
2. Convert BGR → HSV: $\text{HSV} = \text{cv2.cvtColor(BGR, cv2.COLOR\_BGR2HSV)}$
3. Filter by saturation: Keep pixels with $S > 30$ (colored, not grayscale).
4. Compute mean: $\mathbf{f} = \langle H_{\text{mean}}, S_{\text{mean}}, V_{\text{mean}} \rangle$

**K-Means Algorithm:**

Given $n$ players with features $\{\mathbf{f}_1, \ldots, \mathbf{f}_n\}$, partition into 3 clusters:

$$\min_{\mathbf{c}_1, \mathbf{c}_2, \mathbf{c}_3} \sum_{i=1}^{n} \| \mathbf{f}_i - \mathbf{c}_{k(i)} \|^2$$

where $k(i) \in \{1, 2, 3\}$ is the assigned cluster and $\mathbf{c}_j$ is the cluster center.

**Implementation:**
```python
from sklearn.cluster import KMeans
features = np.array([get_jersey_hsv(frame, bbox) for bbox in detections])
kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
kmeans.fit(features)
labels = kmeans.predict(features)
ref_label = np.argmin(np.bincount(labels))  # referee = smallest cluster
```

**Cluster Interpretation:**
- **Cluster 0 & 1:** Teams A and B (largest clusters)
- **Cluster 2 (or whichever is smallest):** Referee

**Hue Wrapping:** Hue is circular (0° ≈ 360° = red), handled by clamping to [0, 180].

---

### **2.4 PlayerTracker: Hungarian Assignment (Greedy Variant)**

**Problem:** Maintain persistent player IDs across frames.

**Input:** 
- Active tracks from frame $t-1$: $\{\mathcal{T}_1, \ldots, \mathcal{T}_m\}$
- New detections in frame $t$: $\{\mathbf{d}_1, \ldots, \mathbf{d}_n\}$ (in board space)

**Cost Matrix:** Euclidean distances

$$C_{i,j} = \| \mathbf{d}_j - \text{predict}(\mathcal{T}_i) \|_2$$

where $\text{predict}(\mathcal{T}_i)$ is the extrapolated position of track $i$.

**Velocity Prediction:**
$$\text{predict}(\mathcal{T}_i) = \mathbf{p}_{t-1} + \mathbf{v}_{t-1}$$

where:
- $\mathbf{p}_{t-1}$: smoothed position at frame $t-1$
- $\mathbf{v}_{t-1}$: estimated velocity = $\mathbf{p}_{t-1} - \mathbf{p}_{t-2}$

**Greedy Hungarian Algorithm:**
1. Sort all pairs $(i, j)$ by cost $C_{i,j}$ (ascending).
2. Iterate through sorted list; assign if both $i$ and $j$ are unassigned.
3. Reject assignments with cost > **150 pixels** (threshold).

**Smoothing (Exponential Moving Average):**
$$\hat{\mathbf{p}}_t = \alpha \mathbf{d}_t + (1 - \alpha) \mathbf{p}_{t-1}$$

where $\alpha = 0.6$ (detection weight).

**Track Lifecycle:**
- **Creation:** Unmatched detection becomes new track.
- **Persistence:** Track survives up to 10 frames without matching (max_frames_missing).
- **Deletion:** Track removed if unmatched for 10+ consecutive frames.

---

### **2.5 BallTracker: Kalman Filter**

**Problem:** Smooth ball position and predict during occlusion.

**State:** $\mathbf{x} = [x, y, v_x, v_y]$ (position + velocity in 2D board space).

**Simplified Implementation:** Apply 1D Kalman filter to each axis independently.

**Kalman Filter Equations (1D per axis):**

**Prediction step:**
$$x_{\text{pred}} = x_{t-1} + v_{t-1} \cdot \Delta t$$
$$p_{\text{pred}} = p_{t-1} + q$$

where:
- $x_{t-1}$: previous smoothed position
- $v_{t-1}$: previous velocity estimate
- $p_{t-1}$: previous uncertainty (covariance)
- $q = 0.01$: process noise (system uncertainty)

**Correction step (if measurement available):**
$$y = z - x_{\text{pred}}$$ (residual / innovation)
$$s = p_{\text{pred}} + r$$ (innovation covariance)
$$k = \frac{p_{\text{pred}}}{s}$$ (Kalman gain)
$$x_{\text{corrected}} = x_{\text{pred}} + k \cdot y$$
$$p_{\text{corrected}} = (1 - k) \cdot p_{\text{pred}}$$

where:
- $z$: measured position
- $r = 4.0$: measurement noise (sensor uncertainty)

**No-Detection Handling:**
- If ball not detected: use prediction without correction.
- Predicts up to **5 frames** without detection.
- Reset if discontinuity > 150 pixels (likely track switch).

---

### **2.6 KPI Calculations**

All KPIs are computed in **metres** using the tactical board pixel-to-metre scale.

**Pixel-to-Metre Conversion:**
$$m_{\text{per\_px\_x}} = \frac{40 \text{ m}}{700 \text{ px}} \approx 0.0571 \text{ m/px}$$
$$m_{\text{per\_px\_y}} = \frac{20 \text{ m}}{350 \text{ px}} \approx 0.0571 \text{ m/px}$$

For position $\mathbf{p}_{\text{px}} = (x, y)$ in pixels:
$$\mathbf{p}_m = (x \cdot m_{\text{per\_px\_x}}, y \cdot m_{\text{per\_px\_y}})$$

#### **2.6.1 Distance Covered**

$$d_{\text{total}} = \sum_{t=1}^{T} \left\| \mathbf{p}_t - \mathbf{p}_{t-1} \right\|_2$$

where $T$ is total frames tracked.

**Anomaly filtering:** If step distance > 10 m (impossible sprint), skip sample.

#### **2.6.2 Top Speed**

$$v_{\max} = \max_{t} \frac{\| \mathbf{p}_t - \mathbf{p}_{t-1} \|_2}{\Delta t}$$

where $\Delta t = 1/\text{fps}$ (frame duration in seconds).

#### **2.6.3 Sprint Count**

A sprint is a continuous period where $v(t) > v_{\text{threshold}}$.

$$\text{sprint\_count} = \#\text{transitions from } v < v_{\text{threshold}} \text{ to } v \geq v_{\text{threshold}}$$

Default threshold: $v_{\text{threshold}} = 5.0$ m/s (typical futsal sprint).

#### **2.6.4 Possession**

Player is in possession if closest to ball and:
$$d_{\text{to\_ball}} < r_{\text{possession}} = 1.5 \text{ m}$$

$$\text{possession\_time} = \sum_{t=1}^{T} \mathbb{1}[\text{closest to ball AND } d < 1.5 \text{ m}] \cdot \frac{1}{\text{fps}}$$

#### **2.6.5 Duel Time**

A duel occurs between two opposing players within duel radius:
$$d_{\text{opponent}} < r_{\text{duel}} = 1.5 \text{ m}$$

$$\text{duel\_frames} = \sum_{t=1}^{T} \mathbb{1}[\exists \text{ opponent within } 1.5 \text{ m}]$$

$$\text{duel\_time} = \frac{\text{duel\_frames}}{\text{fps}}$$

---

## **3. DETAILED TECHNICAL SPECIFICATIONS**

### **3.1 YOLO Detection**

**Model:** YOLOv11n (nano variant)
- **Model size:** ~5.6 MB
- **Speed:** ~25-30 FPS on GPU, ~5-10 FPS on CPU
- **Classes of interest:**
  - Class 0: Person (player)
  - Class 32: Ball (sports ball)
- **Confidence threshold:** 0.3 (default, configurable)

**Bounding Box Filtering:**
1. **Size validation:**
   - Height must be 20-400 pixels (too small = noise, too large = frame edge artifacts).
   - Aspect ratio (width/height) must be in [0.3, 3.0] (too elongated = false positive).

2. **Field boundary:** Feet position must be inside the calibrated 6-point polygon.

3. **Output:** Detections are (x, y, w, h, class_id, confidence).

### **3.2 Tactical Board Rendering**

**Dimensions:** 700 × 350 pixels (40 m × 20 m futsal pitch, 2:1 aspect ratio)

**FIFA Futsal Pitch Markings:**

| Feature | Dimension | Board Position |
|---------|-----------|-----------------|
| Pitch width | 40 m | 0–700 px |
| Pitch height | 20 m | 0–350 px |
| Halfway line | — | x = 350 px |
| Centre circle | 3 m radius | (350, 175) |
| Penalty area (D) | 6 m quarter-circles | x = 0 and x = 700 |
| Goal | 3 m wide | ±1.5 m from center |
| 1st penalty mark | 6 m from goal | 6 m × px_per_m_x from edge |
| 2nd penalty mark | 10 m from goal | 10 m × px_per_m_x from edge |

**Color Scheme:**
- **Pitch:** Dark green (34, 139, 34) BGR
- **Lines:** White (255, 255, 255)
- **Team A:** Blue-ish (255, 80, 80) BGR
- **Team B:** Yellow-ish (80, 255, 255) BGR
- **Referee:** Gray (200, 200, 200) BGR
- **Ball:** Orange (0, 165, 255) BGR

### **3.3 Configuration Parameters**

```python
@dataclass
class Config:
    model_name: str = "yolo11n.pt"
    board_width: int = 700
    board_height: int = 350
    player_class_id: int = 0
    ball_class_id: int = 32
    min_players_for_kmeans: int = 6  # Need 6+ to train team classifier
    yolo_conf_threshold: float = 0.3
    stream_read_retries: int = 30
    stream_read_delay: float = 0.3
    yt_dlp_timeout: int = 30

# KPI Thresholds (in KPITracker)
sprint_threshold_ms: float = 5.0  # m/s
duel_radius_m: float = 1.5        # metres
possession_radius_m: float = 1.5  # metres

# Tracker Thresholds
max_frames_missing: int = 10      # frames before track deletion
max_distance_for_assignment: float = 150  # pixels
ball_max_distance: float = 100    # pixels (discontinuity threshold)
```

---

## **4. DATA FLOW & FILE FORMATS**

### **4.1 Input/Output Pipeline**

```
YouTube URL (or local file)
    ↓
[yt-dlp extraction]
    ↓
video.mp4 (stream URL)
    ↓
[Calibration UI or .npy file]
    ↓
calibration_points.npy (6 × 2 array)
    ↓
[Process frames in headless mode]
    ↓
├─ out/camera.mp4        (annotated camera view)
├─ out/board.mp4         (tactical board video)
├─ out/positions.jsonl   (per-frame positions)
└─ out/kpis.csv          (per-player cumulative stats)
```

### **4.2 JSONL Format (positions.jsonl)**

Each line is a JSON object for one frame:

```json
{
  "frame_idx": 123,
  "timestamp": 4.92,
  "ball": {"x_px": 450.2, "y_px": 175.8, "x_m": 25.7, "y_m": 10.1},
  "players": [
    {"track_id": 1, "x_px": 100.0, "y_px": 200.0, "x_m": 5.7, "y_m": 11.4, "team": 0},
    {"track_id": 2, "x_px": 150.0, "y_px": 250.0, "x_m": 8.6, "y_m": 14.3, "team": 1}
  ]
}
```

### **4.3 CSV Format (kpis.csv)**

| track_id | team | distance_m | top_speed_ms | sprint_count | possession_s | duel_time_s | seen_frames |
|----------|------|-----------|--------------|-------------|------------|-----------|-----------|
| 1 | 0 | 125.4 | 8.2 | 12 | 45.6 | 23.1 | 1200 |
| 2 | 1 | 98.7 | 7.9 | 9 | 38.2 | 31.5 | 1180 |

---

## **5. CALIBRATION INTERFACE**

### **5.1 6-Point Calibration**

Users define 6 control points on a video frame:

1. **TL** (Top-Left): Upper-left corner of field
2. **CT** (Central Top): Halfway line, top edge
3. **TR** (Top-Right): Upper-right corner
4. **BR** (Bottom-Right): Lower-right corner
5. **CB** (Central Bottom): Halfway line, bottom edge
6. **BL** (Bottom-Left): Lower-left corner

These are dragged to match the visible pitch boundary in the camera view.

**UI Features:**
- Points displayed as yellow draggable circles.
- Active point highlighted in blue.
- Real-time preview of the field polygon.
- Press SPACE to confirm and start analysis.

### **5.2 Calibration Persistence**

Saved as `.npy` file (NumPy binary):
```python
np.save("calibration_points.npy", calibration_points)  # (6, 2) array
calibration_points = np.load("calibration_points.npy")
```

---

## **6. PERFORMANCE & CONSTRAINTS**

### **6.1 Computational Requirements**

| Component | GPU (CUDA) | CPU |
|-----------|-----------|-----|
| YOLO inference | ~30 fps (batch=1) | ~5 fps |
| Field mapping | Real-time | Real-time |
| Tracking | Real-time | Real-time |
| K-Means training | ~100 ms | ~200 ms |
| Board rendering | Real-time | Real-time |
| **Overall** | **25-30 fps** | **5-10 fps** |

### **6.2 Memory Usage**

- **YOLO model:** ~5.6 MB on disk, ~200 MB loaded in VRAM
- **Per-frame buffers:** ~10-50 MB depending on resolution
- **KPI tracking:** ~1-10 MB (proportional to player count × duration)

### **6.3 Limitations**

1. **Occlusion:** Tracked players disappear for ≤10 frames, then lost.
2. **Camera zoom/pan:** Recalibrate if camera moves significantly.
3. **Lighting changes:** Team classifier may need retraining (use `--retrain-every` flag).
4. **Ball occlusion:** Kalman filter predicts up to 5 frames; beyond that, ball is lost.
5. **Player count:** Best accuracy with 10-15 visible players; crowded scenes reduce tracking quality.

---

## **7. USAGE EXAMPLES**

### **Interactive Mode:**
```bash
futsal-analytics
# Enter YouTube URL → Calibrate → Analysis starts
```

### **Headless Mode (CI/batch):**
```bash
futsal-analytics \
    --url "https://youtu.be/VIDEO_ID" \
    --calibration calibration.npy \
    --no-gui \
    --device cuda \
    --save-video out/camera.mp4 \
    --save-board-video out/board.mp4 \
    --save-positions out/positions.jsonl \
    --save-kpis out/kpis.csv \
    --max-frames 3000
```

### **Web App:**
```bash
pip install -e ".[viewer]"
streamlit run web/app.py
# Open http://localhost:8501
```

---

## **8. Key Mathematical Insights**

| Formula | Purpose |
|---------|---------|
| $\mathbf{H} \in \mathbb{R}^{3 \times 3}$ (homography) | Projective plane-to-plane transformation |
| $C_{i,j} = \left\| \mathbf{d}_j - \text{predict}(\mathcal{T}_i) \right\|_2$ | Assignment cost in tracking |
| $\hat{\mathbf{p}}_t = 0.6 \mathbf{d}_t + 0.4 \mathbf{p}_{t-1}$ | Position smoothing (EMA) |
| $v(t) = \frac{\left\| \mathbf{p}_t - \mathbf{p}_{t-1} \right\|_2}{1/\text{fps}}$ | Instantaneous speed |
| $d_{\text{total}} = \sum_{t=1}^{T} \left\| \mathbf{p}_t - \mathbf{p}_{t-1} \right\|_2$ | Cumulative distance |

---

This completes the full technical specification of the Futsal Analytics system. The architecture balances real-time performance with tactical accuracy through careful algorithm selection and parameter tuning.
