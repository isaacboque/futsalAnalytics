# Futsal Analytics — Web Viewer

Streamlit dashboard that loads the output files produced by the
`futsal-analytics` CLI and shows interactive overview, heatmaps,
KPI tables, a tactical-board replay, and the recorded videos.

```bash
# Install the viewer extras
pip install -e ".[viewer]"

# Generate output files (do this once per match)
futsal-analytics \
    --url https://www.youtube.com/watch?v=VIDEO_ID \
    --calibration calibration_points.npy \
    --no-gui \
    --save-positions out/positions.jsonl \
    --save-kpis out/kpis.csv \
    --save-board-video out/board.mp4

# Open the viewer
streamlit run web/app.py
```

In the sidebar, point the **Output directory** at the folder containing
`positions.jsonl`, `kpis.csv`, and the optional MP4 files.
