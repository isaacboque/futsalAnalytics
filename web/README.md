# Futsal Analytics — Web App

End-to-end Streamlit app for futsal match analysis. Two pages:

- **Analyse** — paste a YouTube URL, pick a clean frame, place 6 calibration
  points in the browser, and start the analyser. Live log shows progress.
- **Viewer** — interactive Overview, Tracks, Heatmaps, Replay, and Video tabs
  for a single run's outputs.

## Install

```bash
pip install -e ".[viewer]"
```

This pulls in `streamlit`, `plotly`, `pandas`, and
`streamlit-image-coordinates` (used for click-to-place calibration).

## Run

```bash
streamlit run web/app.py
```

A browser tab opens. Use the **sidebar nav** to move between Home / Analyse /
Viewer.

### Analyse flow

1. **Source** — YouTube URL, start time (`MM:SS`), output directory.
2. **Pick a calibration frame** — `Open stream` then `Next frame` / `Skip 30 s`
   / `Skip 60 s` to land on a wide-angle shot that shows the whole pitch.
3. **Calibrate** — click 6 times in order to place `TL → CT → TR → BR → CB → BL`.
   Re-place any point via the right-hand list. `Save calibration` writes
   `cal.npy` into the output directory.
4. **Run** — choose what to record (camera/board videos), compute device, and
   classifier retraining interval. `Start analysis` launches `futsal-analytics`
   as a subprocess with `--no-gui`; stdout streams into the live log. `Stop`
   terminates the run cleanly.

### Viewer flow

Set **Output directory** in the sidebar to a folder containing
`positions.jsonl`, `kpis.csv`, and any optional MP4s. Adjust the **Track
filter** to hide short ID-switch fragments.

## File layout

```
web/
  app.py                  Home page
  _shared.py              CSS, constants, cached loaders, sidebar helpers
  pages/
    1_Analyse.py          Producer page (URL → calibrate → run)
    2_Viewer.py           Consumer page (dashboards, replay, video)
```

## CLI-only? Sure.

The Analyse page is just a friendlier wrapper around the CLI. You can always
do everything from a shell:

```bash
futsal-analytics \
    --url https://www.youtube.com/watch?v=VIDEO_ID \
    --start 17:00 \
    --calibration out/cal.npy \
    --no-gui \
    --save-positions out/positions.jsonl \
    --save-kpis out/kpis.csv \
    --save-video out/camera.mp4 \
    --save-board-video out/board.mp4
```
