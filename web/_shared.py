"""Shared helpers, theme, and constants used across the Streamlit pages."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from futsal_analytics import TacticalBoard

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

DEFAULT_BOARD_W = 700
DEFAULT_BOARD_H = 350

TEAM_COLOR_HEX = {
    0: "#d85c7d",
    1: "#a1a1aa",
    -1: "#52525b",
}
TEAM_LABEL = {0: "Team A", 1: "Team B", -1: "Referee"}
ACCENT = "#d85c7d"


CUSTOM_CSS = """
<style>
section.main > div.block-container {
    padding-top: 1.5rem;
    padding-bottom: 4rem;
    max-width: 1400px;
}

.fa-hero {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 0.25rem;
}
.fa-hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0;
    background: linear-gradient(90deg, #d85c7d 0%, #b84363 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.fa-hero span.tag {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #71717a;
    border: 1px solid #3f3f46;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
}
.fa-meta {
    color: #71717a;
    font-size: 0.85rem;
    margin-top: -0.1rem;
    margin-bottom: 1.25rem;
}

div[data-testid="stMetric"] {
    background: rgba(23, 23, 31, 0.8);
    border: 1px solid #27272a;
    border-radius: 10px;
    padding: 14px 18px;
}
div[data-testid="stMetricLabel"] {
    color: #71717a !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="stMetricValue"] {
    color: #f4f4f5 !important;
    font-size: 1.7rem;
    font-weight: 700;
}

section[data-testid="stSidebar"] .fa-brand {
    border-bottom: 1px solid #27272a;
    padding-bottom: 1rem;
    margin-bottom: 1rem;
}
section[data-testid="stSidebar"] .fa-brand h1 {
    font-size: 1.4rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #d85c7d 0%, #b84363 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
section[data-testid="stSidebar"] .fa-brand p {
    color: #71717a;
    font-size: 0.8rem;
    margin: 0.15rem 0 0;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

.fa-status {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 0.85rem;
    padding: 4px 0;
}
.fa-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.fa-dot.ok    { background: #d85c7d; box-shadow: 0 0 8px #d85c7d60; }
.fa-dot.miss  { background: #3f3f46; }
.fa-status .name { color: #d4d4d8; font-weight: 500; }
.fa-status .meta { color: #52525b; margin-left: auto; font-size: 0.75rem; }

button[data-baseweb="tab"] {
    font-weight: 600;
    letter-spacing: 0.02em;
    padding: 8px 18px !important;
}

.fa-step {
    background: rgba(23, 23, 31, 0.7);
    border: 1px solid #27272a;
    border-radius: 10px;
    padding: 16px 18px;
    height: 100%;
}
.fa-step .num {
    display: inline-block;
    background: #d85c7d;
    color: #0e0e12;
    font-weight: 800;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    text-align: center;
    line-height: 24px;
    margin-bottom: 8px;
}
.fa-step h4 { margin: 0 0 6px 0; color: #f4f4f5; font-size: 1rem; }
.fa-step p  { margin: 0; color: #71717a; font-size: 0.85rem; }
</style>
"""


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system,Segoe UI,Roboto,sans-serif", color="#e7ece8", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
)


def inject_css() -> None:
    """Inject the shared CSS rules into the current page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar_brand() -> None:
    """Render the sidebar brand block (logo + subtitle) used by every page."""
    st.sidebar.markdown(
        '<div class="fa-brand"><h1>Futsal Analytics</h1>'
        '<p>Match analysis suite</p></div>',
        unsafe_allow_html=True,
    )


def styled_plot(fig: go.Figure, **extra) -> go.Figure:
    """Apply the shared Plotly defaults; *extra* overrides any default key."""
    layout = {**PLOTLY_LAYOUT, **extra}
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor="#27272a", zerolinecolor="#27272a")
    fig.update_yaxes(gridcolor="#27272a", zerolinecolor="#27272a")
    return fig


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_positions(path_str: str, mtime: float) -> Tuple[List[dict], int]:
    """Read a JSONL position log, skipping any malformed lines.

    Returns ``(records, n_skipped)``. Survives partial / interleaved writes
    from a crashed or duplicate analyser process \u2014 bad lines are dropped
    rather than crashing the whole viewer.
    """
    del mtime
    path = Path(path_str)
    if not path.exists():
        return [], 0
    records: List[dict] = []
    skipped = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    skipped += 1
    except OSError:
        return [], skipped
    return records, skipped


@st.cache_data(show_spinner=False)
def load_kpis(path_str: str, mtime: float) -> Optional[pd.DataFrame]:
    """Read a KPI CSV, tolerating partial / corrupt rows."""
    del mtime
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, on_bad_lines="skip", engine="python")
    except (pd.errors.EmptyDataError, OSError):
        return None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def pitch_background_png(board_w: int, board_h: int) -> str:
    board = TacticalBoard(board_w, board_h)
    img = board.draw_state([], None)
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


# ---------------------------------------------------------------------------
# Roster + per-player aggregation
# ---------------------------------------------------------------------------
#
# A roster maps every recognised track ID onto a "real" player slot, so
# per-player KPIs aggregate across all fragments of the same physical player
# instead of treating each track ID as a separate player.
#
# Schema of ``out/roster.json``::
#
#     {
#       "teams": {
#         "0": {"label": "Home", "players": [{"id": "h_1", "name": "Player A", "number": 7}, ...]},
#         "1": {"label": "Away", "players": [...]}
#       },
#       "assignments": { "42": "h_1", "17": "h_3", ... }   # track_id -> player_id
#     }
#
# All keys/values are strings so the file is robust to JSON round-tripping.

ROSTER_FILENAME = "roster.json"


def load_roster(path: Path) -> Optional[dict]:
    """Load a roster JSON file. Returns None if the file is missing or invalid."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    data.setdefault("teams", {})
    data.setdefault("assignments", {})
    return data


def save_roster(path: Path, roster: dict) -> None:
    """Atomically write a roster JSON to *path*."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(roster, fp, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def player_label(roster: dict, player_id: str) -> str:
    """Return a 'Name #number' display label for a roster player_id."""
    for team_data in roster.get("teams", {}).values():
        for p in team_data.get("players", []):
            if p.get("id") == player_id:
                name = p.get("name", player_id)
                num = p.get("number")
                return f"{name} #{num}" if num is not None else name
    return player_id


def all_players(roster: dict) -> List[Tuple[str, str, int]]:
    """Flatten roster into a [(player_id, display_label, team_int)] list."""
    out: List[Tuple[str, str, int]] = []
    for team_str, team_data in roster.get("teams", {}).items():
        try:
            team_int = int(team_str)
        except ValueError:
            team_int = -1
        for p in team_data.get("players", []):
            pid = p.get("id")
            if not pid:
                continue
            num = p.get("number")
            label = f"{p.get('name', pid)}" + (f" #{num}" if num is not None else "")
            out.append((pid, label, team_int))
    return out


def apply_roster_to_kpis(kpis_df: pd.DataFrame, roster: Optional[dict]) -> pd.DataFrame:
    """Aggregate per-track KPIs into per-player KPIs using the roster's assignments.

    Returns a new DataFrame indexed by ``player_id`` with the same KPI columns
    summed (or maxed, for top_speed). Unassigned tracks are grouped under
    the player id ``"_unassigned"``.
    """
    if kpis_df is None or kpis_df.empty:
        return pd.DataFrame()

    df = kpis_df.copy()
    assignments = (roster or {}).get("assignments", {}) or {}
    df["player_id"] = df["track_id"].astype(str).map(assignments).fillna("_unassigned")

    grouped = df.groupby("player_id").agg(
        team=("team", "first"),
        distance_m=("distance_m", "sum"),
        top_speed_ms=("top_speed_ms", "max"),
        sprint_count=("sprint_count", "sum"),
        possession_s=("possession_s", "sum"),
        duel_s=("duel_s", "sum"),
        seen_s=("seen_s", "sum"),
        track_count=("track_id", "nunique"),
    ).reset_index()

    if roster is not None:
        grouped["display"] = grouped["player_id"].apply(
            lambda pid: "Unassigned tracks" if pid == "_unassigned" else player_label(roster, pid)
        )
    else:
        grouped["display"] = grouped["player_id"]

    # Round to match the per-track CSV columns
    for col in ("distance_m", "possession_s", "duel_s", "seen_s"):
        grouped[col] = grouped[col].round(2)
    grouped["top_speed_ms"] = grouped["top_speed_ms"].round(2)

    return grouped


def positions_to_dataframe(positions: List[dict]) -> pd.DataFrame:
    rows = []
    for rec in positions:
        for p in rec.get("players", []):
            rows.append(
                {
                    "frame": rec["frame"],
                    "t": rec["t"],
                    "id": p["id"],
                    "team": p["team"],
                    "x": p["x"],
                    "y": p["y"],
                }
            )
    if not rows:
        return pd.DataFrame(columns=["frame", "t", "id", "team", "x", "y"])
    return pd.DataFrame(rows)


def frame_to_players(
    record: dict,
) -> Tuple[List[Tuple[int, np.ndarray, int]], Optional[np.ndarray]]:
    players = [
        (int(p["id"]), np.array([p["x"], p["y"]]), int(p["team"]))
        for p in record.get("players", [])
    ]
    ball = None
    if record.get("ball") is not None:
        b = record["ball"]
        ball = np.array([b["x"], b["y"]])
    return players, ball


# ---------------------------------------------------------------------------
# Sidebar — output directory + file-status badges (shared across pages)
# ---------------------------------------------------------------------------


def output_dir_picker(default: str = "out") -> Path:
    """Render the shared 'Output directory' input + status badges in the sidebar."""
    out_dir_str = st.sidebar.text_input(
        "Output directory",
        value=str(Path(default).absolute()),
        help="Folder containing positions.jsonl, kpis.csv, cal.npy, and optional MP4 videos.",
        key="shared_out_dir",
    )
    out_dir = Path(out_dir_str)

    st.sidebar.markdown("###### Files in this run")
    for label, fname in [
        ("positions.jsonl", "positions.jsonl"),
        ("kpis.csv", "kpis.csv"),
        ("cal.npy", "cal.npy"),
        ("board.mp4", "board.mp4"),
        ("camera.mp4", "camera.mp4"),
    ]:
        _status_badge(label, out_dir / fname)

    return out_dir


def _status_badge(label: str, path: Path) -> None:
    if path.exists():
        size_b = path.stat().st_size
        if size_b >= 1024 * 1024:
            size_str = f"{size_b / 1024 / 1024:.1f} MB"
        elif size_b >= 1024:
            size_str = f"{size_b / 1024:.1f} KB"
        else:
            size_str = f"{size_b} B"
        st.sidebar.markdown(
            f'<div class="fa-status"><span class="fa-dot ok"></span>'
            f'<span class="name">{label}</span><span class="meta">{size_str}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            f'<div class="fa-status"><span class="fa-dot miss"></span>'
            f'<span class="name">{label}</span><span class="meta">missing</span></div>',
            unsafe_allow_html=True,
        )
