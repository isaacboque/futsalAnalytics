"""
Futsal Analytics — Streamlit landing page.

Two-page app:
  - **Analyse**: end-to-end web flow (URL → in-browser 6-point calibration →
    subprocess-based run with live log).
  - **Viewer**: interactive dashboards, heatmaps, replay, and recorded videos
    for a single completed run.

Run:
    pip install -e ".[viewer]"
    streamlit run web/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from _shared import inject_css, render_sidebar_brand

st.set_page_config(
    page_title="Futsal Analytics",
    page_icon=":soccer:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/isaacboque/futsalAnalytics",
        "Report a bug": "https://github.com/isaacboque/futsalAnalytics/issues",
        "About": "Futsal Analytics — match analysis suite.",
    },
)
inject_css()
render_sidebar_brand()


st.markdown(
    '<div class="fa-hero"><h1>Futsal Analytics</h1>'
    '<span class="tag">Home</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="fa-meta">YouTube stream &rarr; calibrated tactical analysis '
    '&rarr; interactive dashboards. End-to-end in the browser.</p>',
    unsafe_allow_html=True,
)


c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown(
        '<div class="fa-step"><span class="num">1</span>'
        '<h4>Analyse</h4>'
        '<p>Open the <strong>Analyse</strong> page in the sidebar. Paste a YouTube URL, '
        'pick a clean calibration frame, click the 6 pitch points, and start the '
        'analyser. A live log shows progress.</p></div>',
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        '<div class="fa-step"><span class="num">2</span>'
        '<h4>View</h4>'
        '<p>Open the <strong>Viewer</strong> page in the sidebar. Set its output '
        'directory to the same folder you analysed into. Browse Overview, Tracks, '
        'Heatmaps, Replay, and recorded videos.</p></div>',
        unsafe_allow_html=True,
    )


st.divider()

st.markdown("##### What gets produced")

st.markdown(
    """
- `cal.npy` — the 6 calibration points used to map camera pixels to the pitch.
- `positions.jsonl` — per-frame player and ball positions in board-pixel space.
- `kpis.csv` — per-track totals: distance, top speed, sprints, possession, duels, seen seconds.
- `camera.mp4` (optional) — the source video with player/ball boxes overlaid.
- `board.mp4` (optional) — the tactical board rendered frame-by-frame.
"""
)

st.markdown("##### Prefer the CLI?")
st.code(
    "futsal-analytics \\\n"
    "  --url <youtube-url> \\\n"
    "  --start 17:00 \\\n"
    "  --calibration out/cal.npy \\\n"
    "  --save-positions out/positions.jsonl \\\n"
    "  --save-kpis out/kpis.csv \\\n"
    "  --save-video out/camera.mp4 \\\n"
    "  --save-board-video out/board.mp4 \\\n"
    "  --device auto --no-gui",
    language="bash",
)

st.caption(
    f"Workspace root: `{Path(__file__).resolve().parents[1]}`"
)
