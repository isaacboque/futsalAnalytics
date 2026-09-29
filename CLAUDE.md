# IAFS (futsal analytics)

Python 3.10+, YOLO (ultralytics) + OpenCV + torch, CUDA-accelerated. Pip-installed
via `pyproject.toml`, no lock file. **GitHub remote is named `futsalAnalytics`, not
`IAFS`** — matters when using `gh` for this repo.

Two subsystems in `src/`: legacy `futsal_analytics/` (the working pipeline) and
newer `pivotiq/` (CLI skeleton only — analysis pipeline is stubbed, don't assume it
does anything yet).

## Commands

- `futsal-analytics` (or `python -m futsal_analytics`) — interactive; or headless:
  `futsal-analytics --url <youtube-url> --calibration calibration_points.npy
  --no-gui --device cuda --max-frames 3000 --save-positions out/positions.jsonl`
- `futsal-calibrate` — standalone calibration UI
- `pivotiq analyze <video.mp4> [--ball-weights ...] [--keypoint-weights ...]`
- `pip install -e ".[viewer]"` then `streamlit run web/app.py --server.address
  0.0.0.0 --server.port 8501` — Streamlit viewer (Analyse/Viewer/Roster pages)
- `python scripts/train.py --data futsal_dataset/futsal_data.yaml [--device cuda]`
- `pytest` — `tests/` (`--import-mode=importlib`) plus `tests/pivotiq/`

`ruff` is fully configured in `pyproject.toml` (line-length 110) and installed in
`.venv`; run `ruff check --fix .` / `ruff format .` before committing.

## Non-obvious things that will bite you

- `torch`/`torchvision` are **not** a direct dependency — they come in transitively
  via `ultralytics`. The installed build here is CUDA 12.6 (`torch==2.14.0+cu126`).
  A plain `pip install -e .` on a fresh machine can silently install CPU-only torch;
  the CUDA wheel needs installing explicitly per pytorch.org.
- `supervision` is pinned `<0.30` — 0.30 removes the `ByteTrack` alias that
  `detection.make_tracker()` depends on. Don't let this pin drift without checking
  that function still works.
- `*.pt`/`yolo*` model weights at the repo root are gitignored and auto-downloaded
  by `ultralytics` on first run (e.g. `yolo11n.pt`) — don't commit them, don't
  assume they're present before a first run.
- `--device {auto,cpu,cuda}` controls YOLO device selection everywhere; `auto` is
  the default.
