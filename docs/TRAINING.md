# Futsal Dataset Format

This is a template for creating your futsal training dataset in YOLO format.

## Directory Structure

```
futsal_dataset/
├── images/
│   ├── train/
│   │   ├── frame_0001.jpg
│   │   ├── frame_0002.jpg
│   │   └── ...
│   └── val/
│       ├── frame_0101.jpg
│       ├── frame_0102.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── frame_0001.txt
│   │   ├── frame_0002.txt
│   │   └── ...
│   └── val/
│       ├── frame_0101.txt
│       ├── frame_0102.txt
│       └── ...
└── futsal_data.yaml
```

## YAML Configuration File

Create `futsal_data.yaml`:

```yaml
# YOLO Dataset Configuration for Futsal

# Dataset root directory (images/ and labels/ are subdirectories)
path: /path/to/futsal_dataset

# Train/val/test splits (relative to path)
train: images/train
val: images/val
test: images/test  # optional

# Number of classes
nc: 80  # 80 for COCO (person=0, ball=32) or fewer for custom

# Class names (must match your annotations)
names:
  0: player
  1: ball
  # ... other classes
```

## Label Format

Each image has a corresponding `.txt` file with the same name. Each line represents one object:

```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: Integer (0 for player, 1 for ball in above example)
- `x_center`, `y_center`, `width`, `height`: Normalized coordinates (0-1)

Example: `frame_0001.txt`
```
0 0.50 0.40 0.15 0.35
0 0.75 0.50 0.12 0.30
1 0.60 0.35 0.05 0.05
```

## Creating Dataset from Videos

### Option 1: Manual Annotation (Recommended for small datasets)
1. Extract frames from futsal videos
2. Use annotation tools:
   - **Roboflow**: https://roboflow.com/ (free tier, supports import/export)
   - **LabelImg**: https://github.com/heartexo/labelImg
   - **Cvat**: https://github.com/openvinotoolkit/cvat

3. Export in YOLO format

### Option 2: Semi-Automatic with Pre-trained YOLO
1. Extract frames
2. Run current `yolo11n.pt` to get initial detections
3. Manually correct annotations
4. Fine-tune on corrected data

### Option 3: Python Script to Extract Frames

```python
import cv2
from pathlib import Path

video_path = "futsal_game.mp4"
output_dir = "futsal_dataset/images/train"
Path(output_dir).mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0
skip_frames = 5  # Extract every 5th frame to reduce redundancy

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_count % skip_frames == 0:
        filename = f"{output_dir}/frame_{frame_count:04d}.jpg"
        cv2.imwrite(filename, frame)
    
    frame_count += 1

cap.release()
print(f"Extracted {frame_count // skip_frames} frames")
```

## Dataset Tips for Futsal

- **Minimum frames**: 100-200 per split (train/val)
- **Frame diversity**: Include different:
  - Court positions
  - Lighting conditions
  - Jersey colors (both teams)
  - Ball positions (ground, air, occluded)
- **Player annotations**: Include full body when possible
- **Ball**: Always annotate visible ball, even if small
- **Train/val split**: 80/20 is typical

## Using the Training Script

```bash
# Install dependencies
pip install ultralytics

# Basic training (nano model, 50 epochs)
python scripts/train.py --data futsal_dataset/futsal_data.yaml

# Larger model with GPU (faster)
python scripts/train.py --data futsal_dataset/futsal_data.yaml \
  --model yolo11m.pt --batch 32

# Resume previous training
python scripts/train.py --data futsal_dataset/futsal_data.yaml --resume

# CPU only (if no GPU)
python scripts/train.py --data futsal_dataset/futsal_data.yaml \
  --device cpu --batch 4 --epochs 30
```

## Results

After training, best model saved to: `runs/detect/runs/detect/futsal_train-7/weights/best.pt`

View training plots: `runs/detect/runs/detect/futsal_train-7/results.png`

## Using Trained Model

```python
from ultralytics import YOLO

model = YOLO("runs/detect/runs/detect/futsal_train-7/weights/best.pt")
results = model("video.mp4", conf=0.3)
```
