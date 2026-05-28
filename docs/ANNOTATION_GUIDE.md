# Annotation Guide - Next Steps

## ✅ What You Have
- **300 frames** extracted from YouTube futsal video
- Location: `futsal_dataset/images/all/`
- Ready for annotation

## 📝 How to Annotate

### **Option 1: Roboflow (EASIEST - Recommended)**

1. **Sign up** (free): https://roboflow.com
2. **Create new project**:
   - Click "Create New Project"
   - Name: "Futsal"
   - Task: "Object Detection"
   - License: "Public"
3. **Upload images**:
   - Open your project
   - Drag & drop folder: `futsal_dataset/images/all/`
4. **Annotate**:
   - Click on images one by one
   - Draw boxes around **players** (class: "player")
   - Draw boxes around **ball** (class: "ball")
   - Hit spacebar to move to next image
5. **Export**:
   - Click "Export"
   - Format: **"YOLO v11"**
   - Download zip file
   - Unzip contents to: `futsal_dataset/`
   - Structure will auto-organize into train/val/test

**Time estimate**: 10-20 minutes for 300 frames (use fast speed)

---

### **Option 2: LabelImg (Desktop App)**

1. **Install**:
   ```bash
   pip install labelImg
   ```

2. **Run**:
   ```bash
   labelImg "futsal_dataset/images/all"
   ```

3. **Setup**:
   - Click "PascalVOC" → Change to "YOLO"
   - Click "Edit" → "Edit Classes" → Add:
     ```
     player
     ball
     ```

4. **Annotate**:
   - Click "Create RectBox"
   - Draw box around object
   - Select class from dropdown
   - Click "Save" or press "Ctrl+S"

5. **Export**:
   - Create `futsal_dataset/labels/all/` directory
   - Files auto-save as `.txt` alongside images

---

## 📊 After Annotation

Once you've annotated your frames:

```bash
# 1. Verify labels exist
ls futsal_dataset/labels/all/  # Should have .txt files

# 2. Split into train/val (if not done by Roboflow)
python scripts/split_dataset.py --dataset futsal_dataset

# 3. Train YOLO
python scripts/train.py --data futsal_dataset/futsal_dataset.yaml
```

---

## ⚡ Quick Tips

- **You don't need to annotate all 300 frames** - start with 100 for testing
- **Roboflow is 10x faster** than desktop apps (UI designed for speed)
- **Annotation time**: ~30-50 seconds per frame with boxes
- **Quality > Quantity**: Better to carefully annotate 100 frames than poorly annotate 300
- **Focus on diverse frames**: Different court areas, lighting, player positions

---

## 🎯 Annotation Checklist

For each frame:
- [ ] Find all **visible players** → draw tight boxes
- [ ] Find **ball** (even if small) → draw box
- [ ] Assign correct class (player or ball)
- [ ] Move to next frame

---

## 📍 Current Structure

```
futsal_dataset/
├── images/
│   └── all/
│       ├── frame_00001.jpg  ← Ready to annotate
│       ├── frame_00002.jpg
│       └── ... (300 total)
├── labels/
│   └── all/
│       └── (empty - will be filled during annotation)
└── futsal_dataset.yaml
```

---

## 💡 What Happens Next

1. **Annotate frames** in Roboflow or LabelImg
2. **Split dataset** using the split script
3. **Train model** with the train script
4. **Use trained model** in your futsal analyzer

Once trained, update your config:
```python
config = Config(
    model_name="runs/detect/runs/detect/futsal_train-7/weights/best.pt",
    yolo_conf_threshold=0.3
)
```

---

**Ready? Start with Roboflow:** https://roboflow.com
