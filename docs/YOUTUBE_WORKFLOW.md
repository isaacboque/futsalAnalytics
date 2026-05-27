# YouTube to YOLO Training - Complete Workflow

Extract futsal videos from YouTube, annotate them, and train YOLO11 on your own data.

## Prerequisites

Install required tools:

```bash
# Python dependencies
pip install yt-dlp ultralytics

# FFmpeg (required for frame extraction)
# Windows: choco install ffmpeg
# Mac: brew install ffmpeg
# Linux: sudo apt-get install ffmpeg
# Or download from: https://ffmpeg.org/download.html
```

## Step 1: Extract Frames from YouTube

```bash
# Basic extraction (2 fps, all frames)
python scripts/extract_frames.py \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output futsal_dataset/images/all

# Extract from specific time range
python scripts/extract_frames.py \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output futsal_dataset/images/all \
  --start 00:05:00 \
  --end 00:15:00

# Higher fps with frame skipping (reduces redundancy)
python scripts/extract_frames.py \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output futsal_dataset/images/all \
  --fps 5 \
  --skip-frames 3

# Use local video (faster, no download)
python scripts/extract_frames.py \
  --video my_futsal_game.mp4 \
  --output futsal_dataset/images/all \
  --fps 2
```

**Expected output:**
```
futsal_dataset/
├── images/
│   └── all/
│       ├── frame_00001.jpg
│       ├── frame_00002.jpg
│       └── ...
└── labels/
    └── all/
        (empty, will be filled during annotation)
```

## Step 2: Annotate Frames

You need to draw bounding boxes around **players** and **ball** in each frame.

### Option A: Roboflow (Easiest - Web-based, Free)

1. Go to https://roboflow.com
2. Create free account and new project
3. Upload your frames from `futsal_dataset/images/all/`
4. Annotate frames online (draw boxes around players and ball)
5. **Export as "YOLO v11"** format
6. Unzip to: `futsal_dataset/`

### Option B: LabelImg (Desktop App)

1. Install: `pip install labelImg`
2. Run: `labelImg futsal_dataset/images/all`
3. Set class names: `player`, `ball`
4. Draw boxes around players and ball
5. Export as YOLO format to `futsal_dataset/labels/all/`

### Option C: CVAT (Professional, Web-based)

https://github.com/openvinotoolkit/cvat
- More features, steeper learning curve
- Best for large projects

**After annotation, structure should be:**
```
futsal_dataset/
├── images/
│   └── all/
│       ├── frame_00001.jpg
│       ├── frame_00002.jpg
│       └── ...
└── labels/
    └── all/
        ├── frame_00001.txt
        ├── frame_00002.txt
        └── ...
```

**Label format** (each line = one object):
```
0 0.50 0.40 0.15 0.30
1 0.65 0.35 0.04 0.04
```
- `0` = player, `1` = ball
- Values are normalized (0-1): `x_center y_center width height`

## Step 3: Split into Train/Val

Organize frames into training and validation sets:

```bash
python scripts/split_dataset.py --dataset futsal_dataset

# Custom split (80/10/10)
python scripts/split_dataset.py --dataset futsal_dataset --split 0.8 0.1 0.1
```

**Result:**
```
futsal_dataset/
├── images/
│   ├── train/
│   │   ├── frame_00001.jpg
│   │   ├── frame_00002.jpg
│   │   └── ...
│   └── val/
│       ├── frame_00101.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── frame_00001.txt
│   │   └── ...
│   └── val/
│       ├── frame_00101.txt
│       └── ...
└── futsal_dataset.yaml  (auto-created)
```

## Step 4: Train YOLO

```bash
# Basic training (nano model, 50 epochs)
python scripts/train.py --data futsal_dataset/futsal_dataset.yaml

# Larger model with GPU
python scripts/train.py --data futsal_dataset/futsal_dataset.yaml \
  --model yolo11m.pt \
  --batch 32 \
  --epochs 100

# CPU only (slow, for testing)
python scripts/train.py --data futsal_dataset/futsal_dataset.yaml \
  --device cpu \
  --batch 4 \
  --epochs 30

# Resume interrupted training
python scripts/train.py --data futsal_dataset/futsal_dataset.yaml --resume
```

**Training output:**
```
runs/detect/futsal_train/
├── weights/
│   ├── best.pt       ← Your trained model
│   └── last.pt
├── results.csv
└── results.png       ← Accuracy curves
```

## Step 5: Use Your Trained Model

Update your code to use the new model:

```python
from ultralytics import YOLO

# Use your trained model instead of yolo11n.pt
model = YOLO("runs/detect/futsal_train/weights/best.pt")

# Detect in video or image
results = model("my_video.mp4", conf=0.3)
```

Or update config:

```python
from futsal_analytics.config import Config

config = Config(
    model_name="runs/detect/futsal_train/weights/best.pt",
    yolo_conf_threshold=0.3
)
```

---

## Minimum Dataset Size

| Model | Min Frames | Training Time |
|-------|-----------|---|
| yolo11n (nano) | 100-200 | 5-10 min (GPU) / 30+ min (CPU) |
| yolo11s (small) | 150-300 | 15-30 min (GPU) |
| yolo11m (medium) | 300-500 | 1-2 hours (GPU) |

**Start small** (100 frames) to test the pipeline, then add more.

---

## Tips for Better Results

1. **Diverse frames**: Include:
   - Different parts of the court
   - Different lighting conditions
   - Different player positions
   - Ball in different locations
   - Both teams

2. **Accurate annotations**:
   - Tight boxes around players (include full body)
   - Always mark the ball, even if small or partially occluded
   - Use consistent sizing

3. **Training tips**:
   - Start with `yolo11n` for fast iteration
   - Use early stopping (`--patience 10`) to avoid overfitting
   - Increase epochs if validation loss is still decreasing
   - If VRAM limited, reduce `--batch` size

4. **Quality check**:
   - After training, run inference on test set
   - Check detection quality visually
   - If poor, add more annotated frames and retrain

---

## Troubleshooting

### FFmpeg not found
```bash
# Download from https://ffmpeg.org/download.html
# Or install via package manager

# Verify installation
ffmpeg -version
```

### No labels found for images
- Ensure labels are in `labels/all/` with same filename as images
- Example: `frame_00001.jpg` → `frame_00001.txt`

### Out of GPU memory
```bash
# Reduce batch size
python scripts/train.py --data futsal_dataset/futsal_dataset.yaml --batch 8
```

### Training is too slow
```bash
# Use smaller model or GPU
python scripts/train.py --data futsal_dataset/futsal_dataset.yaml \
  --model yolo11n.pt --device 0
```

### Low detection accuracy
- Add more annotated frames (200+ minimum)
- Check annotation quality
- Try larger model (`yolo11m`)
- Increase training epochs

---

## Next Steps

1. Find a futsal YouTube video
2. Run frame extraction
3. Annotate in Roboflow (easiest)
4. Split dataset
5. Train model
6. Test on your analysis pipeline

Questions? Check [TRAINING.md](TRAINING.md) for more details.
