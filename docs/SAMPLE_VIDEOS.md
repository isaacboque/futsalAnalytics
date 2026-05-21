# Sample sources for Futsal Analytics

## Supported input formats

- **YouTube URLs** — public videos and live streams, resolved via `yt-dlp`.
  ```
  https://www.youtube.com/watch?v=VIDEO_ID
  https://www.youtube.com/live/STREAM_ID
  https://youtu.be/VIDEO_ID
  ```
- **Local video files** — `.mp4`, `.mov`, `.mkv`, `.avi` (anything OpenCV
  can decode via ffmpeg). Passed as `--url` on the CLI or uploaded directly
  through the Analyse page.

The CLI auto-detects local-file paths and skips yt-dlp for them.

---

## Two ways to run an analysis

### A) Web app (Analyse → Viewer)

```bash
pip install -e ".[viewer]"
streamlit run web/app.py
```

Open http://localhost:8501, go to **Analyse**:

1. Choose **YouTube URL** or **Local file**.
2. Set start time as `HH:MM:SS`, `MM:SS`, or seconds.
3. Click **Open stream**, then skip forward to a wide-angle calibration frame.
4. Click the 6 pitch points (`TL → CT → TR → BR → CB → BL`).
5. Pick the run options (camera/board recording, device, inference size,
   frame stride) and hit **Start analysis**.

Live previews update every ~½ second. Switch to the **Viewer** page when the
run completes (or even while it's running — the Overview tab populates
progressively).

### B) CLI

```bash
# Quick interactive (prompts for URL)
futsal-analytics

# Fully scripted
futsal-analytics \
    --url https://www.youtube.com/watch?v=VIDEO_ID \
    --start 17:00 \
    --calibration out/cal.npy \
    --no-gui \
    --save-positions out/positions.jsonl \
    --save-kpis out/kpis.csv \
    --save-video out/camera.mp4 \
    --save-board-video out/board.mp4 \
    --device auto --imgsz 640 --frame-stride 1
```

---

## Tips for choosing a good source

- **Best results**: fixed side-camera, ~150° field of view — the standard
  futsal broadcast angle. Multi-camera and pan-zoom-tilt footage will
  confuse calibration.
- **Resolution**: 720p–1080p is the sweet spot. Above 1080p, YOLO is
  internally resized to `--imgsz` anyway; below 480p, ball detection
  degrades noticeably.
- **Live streams**: pass `--skip-when-behind 2.0` so the analyser drops
  frames if it falls more than 2 s behind realtime. Combine with
  `--frame-stride 2` and `--imgsz 480` for CPU-only setups.
- **First few seconds**: many broadcasts start with sponsor logos or
  replays. Use `--start` to skip past them, or use the in-browser frame
  picker.

---

## Where outputs go

The Analyse page writes everything into a single output directory
(default `out/`):

| File | Produced by | Used by |
|---|---|---|
| `cal.npy` | calibration step | every future run for the same camera |
| `positions.jsonl` | per-frame logger | Viewer heatmaps + replay |
| `kpis.csv` | flushed every 30 frames | Viewer Overview + Tracks |
| `camera.mp4` | annotated source | Viewer Video tab |
| `board.mp4` | tactical board | Viewer Video tab |
| `_live_camera.jpg` / `_live_board.jpg` | snapshots every N frames | Live previews in Analyse |
| `.run.lock` | startup → exit | prevents accidental dual-runs |
| `_source.<ext>` | only for local uploads | the analyser subprocess |
