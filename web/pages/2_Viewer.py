"""
Viewer page — interactive dashboard for a single match run.

Loads the artefacts produced by the ``futsal-analytics`` CLI (or the Analyse
page) from an output directory and renders dashboards, heatmaps and a
tactical-board replay.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `_shared` import

from _shared import (
    ACCENT,
    DEFAULT_BOARD_H,
    DEFAULT_BOARD_W,
    PLOTLY_LAYOUT,
    ROSTER_FILENAME,
    TEAM_COLOR_HEX,
    TEAM_LABEL,
    apply_roster_to_kpis,
    file_mtime,
    frame_to_players,
    inject_css,
    load_kpis,
    load_positions,
    load_roster,
    output_dir_picker,
    pitch_background_png,
    positions_to_dataframe,
    render_sidebar_brand,
    styled_plot,
)

from futsal_analytics import TacticalBoard

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------


st.set_page_config(
    page_title="Viewer · Futsal Analytics",
    page_icon=":soccer:",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
render_sidebar_brand()


# ---------------------------------------------------------------------------
# Sidebar — output directory + track filter
# ---------------------------------------------------------------------------


out_dir = output_dir_picker(default="out")
positions_path = out_dir / "positions.jsonl"
kpis_path = out_dir / "kpis.csv"
board_video_path = out_dir / "board.mp4"
camera_video_path = out_dir / "camera.mp4"
roster_path = out_dir / ROSTER_FILENAME

positions, n_bad_positions = load_positions(str(positions_path), file_mtime(positions_path))
kpis_df_raw = load_kpis(str(kpis_path), file_mtime(kpis_path))
positions_df_raw = positions_to_dataframe(positions)
roster = load_roster(roster_path)

if n_bad_positions > 0:
    st.warning(
        f"Skipped **{n_bad_positions}** malformed lines in `positions.jsonl`. "
        "This usually means two analyser processes wrote into the same output "
        "directory concurrently \u2014 only the surviving lines are shown below.",
        icon=":material/warning:",
    )

st.sidebar.markdown("###### Track filter")
min_track_seconds = st.sidebar.slider(
    "Hide tracks shorter than (s)",
    min_value=0.0,
    max_value=15.0,
    value=2.0,
    step=0.5,
    help=(
        "Tracks shorter than this many seconds are likely ID-switch fragments "
        "rather than real player coverage."
    ),
)

if kpis_df_raw is not None and not kpis_df_raw.empty:
    kept_ids = set(
        kpis_df_raw.loc[kpis_df_raw["seen_s"] >= min_track_seconds, "track_id"].tolist()
    )
    kpis_df = kpis_df_raw[kpis_df_raw["track_id"].isin(kept_ids)].reset_index(drop=True)
else:
    kept_ids = (
        set(positions_df_raw["id"].unique().tolist())
        if not positions_df_raw.empty
        else set()
    )
    kpis_df = kpis_df_raw

if not positions_df_raw.empty and kept_ids:
    positions_df = positions_df_raw[positions_df_raw["id"].isin(kept_ids)].reset_index(drop=True)
else:
    positions_df = positions_df_raw

n_raw_tracks = (
    int(kpis_df_raw["track_id"].nunique())
    if kpis_df_raw is not None and not kpis_df_raw.empty
    else (int(positions_df_raw["id"].nunique()) if not positions_df_raw.empty else 0)
)
n_filtered_tracks = (
    int(kpis_df["track_id"].nunique())
    if kpis_df is not None and not kpis_df.empty
    else (int(positions_df["id"].nunique()) if not positions_df.empty else 0)
)
hidden_tracks = max(0, n_raw_tracks - n_filtered_tracks)
st.sidebar.caption(
    f"Showing **{n_filtered_tracks}** of {n_raw_tracks} tracks "
    f"({hidden_tracks} hidden)"
)

# ---------------------------------------------------------------------------
# Optional roster aggregation
# ---------------------------------------------------------------------------
aggregate_by_player = False
n_assigned = 0
if roster is not None:
    n_assigned = sum(
        1 for tid in (kpis_df["track_id"].astype(str).tolist()
                      if kpis_df is not None and not kpis_df.empty else [])
        if tid in roster.get("assignments", {})
    )
    st.sidebar.markdown("###### Roster")
    st.sidebar.caption(
        f"`roster.json` loaded \u2014 {n_assigned}/{n_filtered_tracks} tracks assigned."
    )
    aggregate_by_player = st.sidebar.toggle(
        "Aggregate by player",
        value=True,
        help=(
            "Roll up per-track KPIs to one row per assigned player. "
            "Unassigned tracks are grouped together."
        ),
    )
else:
    st.sidebar.markdown("###### Roster")
    st.sidebar.caption(
        "No `roster.json` in this folder. Open the **Roster** page to assign "
        "tracks to real players."
    )

agg_kpis_df = None
if aggregate_by_player and kpis_df is not None and not kpis_df.empty:
    agg_kpis_df = apply_roster_to_kpis(kpis_df, roster)
    # Replace kpis_df so all downstream tabs render per-player automatically.
    # ``track_id`` here is actually the player display label (string).
    if agg_kpis_df is not None and not agg_kpis_df.empty:
        kpis_df = agg_kpis_df.rename(columns={"display": "track_id"}).drop(
            columns=["player_id"], errors="ignore",
        )


def _format_track(i) -> str:
    """Display label for a track / player id, works for both int and str."""
    try:
        return f"#{int(i)}"
    except (TypeError, ValueError):
        return str(i)


def _classify_mp4(path: Path) -> str:
    """Return one of: 'h264', 'mp4v', 'incomplete', 'unknown'.

    cv2.VideoWriter writes the `moov` (metadata) atom only on .release(),
    so a video from a killed/crashed analyser has just ftyp + mdat and no
    codec info anywhere. Browsers and VLC both refuse such files; we want
    to tell the user clearly instead of showing an opaque "MIME type"
    error from the HTML5 video element.

    The codec FourCC (`avc1` for H.264, `mp4v` for MP4 Visual) lives in
    the `stsd` sample-description atom inside `moov`, which OpenCV writes
    at the tail of the file. We scan both head and tail to be safe.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as fp:
            head = fp.read(8192)
            if size > 8192:
                fp.seek(max(0, size - 131_072))
                tail = fp.read()
            else:
                tail = b""
    except OSError:
        return "unknown"

    combined = head + tail
    if b"moov" not in combined:
        return "incomplete"
    if b"avc1" in combined or b"avcC" in combined:
        return "h264"
    if b"mp4v" in combined:
        return "mp4v"
    return "unknown"


def _render_video_block(label: str, path: Path) -> None:
    st.markdown(f"##### {label}")
    kind = _classify_mp4(path)
    size_mb = path.stat().st_size / 1_048_576

    if kind == "h264":
        st.video(str(path))
    elif kind == "mp4v":
        st.warning(
            "This MP4 uses the `mp4v` codec, which most browsers refuse to "
            "play. The file itself is intact \u2014 download it and open in "
            "VLC, or re-run the analyser to get a browser-friendly "
            "`avc1` / H.264 file.",
            icon=":material/movie:",
        )
    elif kind == "incomplete":
        st.error(
            f"`{path.name}` is **structurally incomplete** \u2014 the analyser "
            "was killed before it could write the MP4 trailer (`moov` atom). "
            "Neither browsers nor VLC will play this file. Delete it and run "
            "a fresh analysis.",
            icon=":material/broken_image:",
        )
    else:
        st.warning(
            "Could not determine the codec of this file. Try the download "
            "link and open it in VLC.",
            icon=":material/help:",
        )

    try:
        with path.open("rb") as fp:
            st.download_button(
                f"Download {path.name} ({size_mb:.1f} MB)",
                data=fp,
                file_name=path.name,
                mime="video/mp4",
                key=f"dl_{path.name}",
            )
    except OSError:
        st.caption(f"`{path.name}` is locked (the analyser may still be writing it).")


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------


if not positions and (kpis_df is None or kpis_df.empty):
    st.markdown(
        '<div class="fa-hero"><h1>Futsal Analytics</h1>'
        '<span class="tag">Viewer</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="fa-meta">No match data found in the selected output directory.</p>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(
            '<div class="fa-step"><span class="num">1</span>'
            '<h4>Run an analysis</h4>'
            '<p>Open the <strong>Analyse</strong> page in the sidebar and feed in a YouTube '
            'URL. It calibrates the pitch in-browser and runs the analyser for you.</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="fa-step"><span class="num">2</span>'
            '<h4>Point the viewer here</h4>'
            '<p>Set the <strong>Output directory</strong> in the sidebar to the run\u2019s '
            'output folder. The dashboards refresh automatically.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.code(
        "futsal-analytics \\\n"
        "  --url <youtube-url> \\\n"
        "  --calibration cal.npy \\\n"
        "  --no-gui \\\n"
        "  --save-positions out/positions.jsonl \\\n"
        "  --save-kpis out/kpis.csv \\\n"
        "  --save-board-video out/board.mp4",
        language="bash",
    )
    st.stop()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


duration_s = max((r["t"] for r in positions), default=0.0)
n_frames = len(positions)

st.markdown(
    '<div class="fa-hero"><h1>Futsal Analytics</h1>'
    '<span class="tag">Viewer</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p class="fa-meta">Loaded from <code>{out_dir.name}</code> &middot; '
    f'{n_frames:,} frames &middot; {duration_s:.0f}s &middot; '
    f'{n_filtered_tracks} tracks (of {n_raw_tracks})</p>',
    unsafe_allow_html=True,
)

with st.expander("About \u201ctracks\u201d vs. \u201cplayers\u201d", expanded=False):
    st.markdown(
        "Each **track ID** is one continuous detection of a player. The motion-only "
        "tracker (ByteTrack) starts a fresh ID whenever a player is occluded, leaves "
        "the camera frame, or is re-acquired after lighting changes, so the same "
        "physical player typically appears under several IDs across a match.\n\n"
        "- **Team-level totals on the Overview tab are reliable** — they sum across "
        "all tracks of a team and ID switches don\u2019t change the totals.\n"
        "- **Per-track views (Tracks, Heatmaps Per-track) are directional, not "
        "literal** \u2014 they describe one continuous detection window, not a full "
        "player\u2019s match.\n"
        "- Use the sidebar **track filter** to hide short fragments below a chosen "
        "duration threshold."
    )

_tracks_tab_label = "Players" if aggregate_by_player else "Tracks"
tab_overview, tab_players, tab_heatmaps, tab_replay, tab_video = st.tabs(
    ["Overview", _tracks_tab_label, "Heatmaps", "Replay", "Video"]
)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


with tab_overview:
    if kpis_df is None or kpis_df.empty:
        st.warning("kpis.csv not found — overview disabled.")
    else:
        total_distance = float(kpis_df["distance_m"].sum())
        max_top_speed = float(kpis_df["top_speed_ms"].max())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frames", f"{n_frames:,}")
        c2.metric("Duration", f"{duration_s:.0f} s")
        c3.metric("Total distance", f"{total_distance:,.0f} m")
        c4.metric("Top speed", f"{max_top_speed:.1f} m/s")

        st.divider()

        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            st.markdown("##### Distance per team")
            team_dist = kpis_df.groupby("team")["distance_m"].sum().reset_index()
            team_dist["label"] = team_dist["team"].map(TEAM_LABEL).fillna("Unknown")
            fig = px.bar(
                team_dist,
                x="label",
                y="distance_m",
                color="label",
                color_discrete_map={
                    TEAM_LABEL[0]: TEAM_COLOR_HEX[0],
                    TEAM_LABEL[1]: TEAM_COLOR_HEX[1],
                    TEAM_LABEL[-1]: TEAM_COLOR_HEX[-1],
                },
                labels={"label": "", "distance_m": "Distance (m)"},
                text="distance_m",
            )
            fig.update_traces(texttemplate="%{text:.0f} m", textposition="outside")
            styled_plot(fig, height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("##### Possession share")
            poss = (
                kpis_df[kpis_df["possession_s"] > 0]
                .groupby("team")["possession_s"]
                .sum()
                .reset_index()
            )
            if poss.empty:
                st.info("No possession events recorded.")
            else:
                poss["label"] = poss["team"].map(TEAM_LABEL).fillna("Unknown")
                fig = px.pie(
                    poss,
                    values="possession_s",
                    names="label",
                    color="label",
                    color_discrete_map={
                        TEAM_LABEL[0]: TEAM_COLOR_HEX[0],
                        TEAM_LABEL[1]: TEAM_COLOR_HEX[1],
                        TEAM_LABEL[-1]: TEAM_COLOR_HEX[-1],
                    },
                    hole=0.55,
                )
                fig.update_traces(textinfo="label+percent")
                styled_plot(fig, height=320, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("##### Top-speed leaderboard")
        st.caption(
            "Speeds belong to individual tracks, not necessarily distinct players."
        )
        leaderboard = kpis_df.nlargest(10, "top_speed_ms").copy()
        leaderboard["Team"] = leaderboard["team"].map(TEAM_LABEL).fillna("Unknown")
        leaderboard["label"] = leaderboard["track_id"].apply(_format_track)
        fig = px.bar(
            leaderboard,
            x="top_speed_ms",
            y="label",
            color="Team",
            color_discrete_map={
                TEAM_LABEL[0]: TEAM_COLOR_HEX[0],
                TEAM_LABEL[1]: TEAM_COLOR_HEX[1],
                TEAM_LABEL[-1]: TEAM_COLOR_HEX[-1],
            },
            orientation="h",
            labels={"top_speed_ms": "Top speed (m/s)", "label": "Track"},
            text="top_speed_ms",
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_yaxes(autorange="reversed", type="category")
        styled_plot(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------


with tab_players:
    if kpis_df is None or kpis_df.empty:
        st.warning("kpis.csv not found or all rows filtered out.")
    else:
        if aggregate_by_player:
            st.success(
                f"Per-player KPIs: rolled up across the {n_assigned} track(s) "
                "assigned to each player via the **Roster** page.",
                icon=":material/check_circle:",
            )
        else:
            st.info(
                "These rows are **per-track**, not per-player. The same physical player "
                "may appear in several rows. Open the **Roster** page to assign tracks "
                "to real players, or use the sidebar filter to hide short fragments.",
                icon=":material/info:",
            )

        display_df = kpis_df.copy()
        display_df["Team"] = display_df["team"].map(TEAM_LABEL).fillna("Unknown")
        display_df = display_df[
            ["track_id", "Team", "distance_m", "top_speed_ms",
             "sprint_count", "possession_s", "duel_s", "seen_s"]
        ].sort_values("seen_s", ascending=False).reset_index(drop=True)

        st.markdown(
            "##### All players" if aggregate_by_player else "##### All tracks (filtered)"
        )
        st.dataframe(
            display_df,
            use_container_width=True,
            height=320,
            hide_index=True,
            column_config={
                "track_id": (
                    st.column_config.TextColumn("Player", width="medium")
                    if aggregate_by_player else
                    st.column_config.NumberColumn("Track", width="small")
                ),
                "Team": st.column_config.TextColumn("Team", width="small"),
                "seen_s": st.column_config.ProgressColumn(
                    "Seen (s)",
                    min_value=0,
                    max_value=float(kpis_df["seen_s"].max() or 1),
                    format="%.1f s",
                ),
                "distance_m": st.column_config.NumberColumn("Distance (m)", format="%.1f"),
                "top_speed_ms": st.column_config.NumberColumn("Top speed (m/s)", format="%.2f"),
                "sprint_count": st.column_config.NumberColumn("Sprints"),
                "possession_s": st.column_config.NumberColumn("Possession (s)", format="%.1f"),
                "duel_s": st.column_config.NumberColumn("Duels (s)", format="%.1f"),
            },
        )

        st.divider()
        st.markdown("##### Track vs team average")
        st.caption(
            "Compares one continuous-detection window to the average across all "
            "tracks of the same team."
        )

        col_pick, col_radar = st.columns([1, 3])
        with col_pick:
            player_ids = sorted(kpis_df["track_id"].astype(str).unique().tolist())
            selected_id = st.selectbox(
                "Player" if aggregate_by_player else "Track",
                player_ids,
                format_func=_format_track,
            )
            player_row = kpis_df[kpis_df["track_id"].astype(str) == str(selected_id)].iloc[0]
            team = int(player_row["team"])
            st.metric("Team", TEAM_LABEL.get(team, "Unknown"))
            st.metric("Seen", f"{player_row['seen_s']:.1f} s")
            st.metric("Distance", f"{player_row['distance_m']:.1f} m")
            st.metric("Top speed", f"{player_row['top_speed_ms']:.2f} m/s")

        with col_radar:
            metric_cols = ["distance_m", "top_speed_ms", "sprint_count",
                           "possession_s", "duel_s", "seen_s"]
            labels = ["Distance", "Top speed", "Sprints", "Possession", "Duels", "Seen"]

            team_avg = (
                kpis_df[kpis_df["team"] == team][metric_cols].mean().values.astype(float)
            )
            player_vals = player_row[metric_cols].values.astype(float)

            # Percentile-rank normalisation across the (filtered) squad.
            # Each metric maps to its empirical CDF, so axes carry equal
            # weight regardless of unit and a single dominant track (e.g.
            # very high "Seen") doesn't squash the rest of the chart.
            def _percentile_norm(values: np.ndarray, target: float) -> float:
                if values.size == 0:
                    return 0.0
                # Fraction of squad with value <= target
                return float((values <= target).mean())

            squad_values = kpis_df[metric_cols].values.astype(float)
            player_norm = np.array([
                _percentile_norm(squad_values[:, i], player_vals[i])
                for i in range(len(metric_cols))
            ])
            team_norm = np.array([
                _percentile_norm(squad_values[:, i], team_avg[i])
                for i in range(len(metric_cols))
            ])

            fig = go.Figure()
            entity = "Player" if aggregate_by_player else "Track"
            fig.add_trace(
                go.Scatterpolar(
                    r=np.concatenate([player_norm, [player_norm[0]]]),
                    theta=labels + [labels[0]],
                    fill="toself",
                    name=f"{entity} {_format_track(selected_id)}",
                    line=dict(color=ACCENT, width=2),
                    fillcolor="rgba(216,92,125,0.2)",
                )
            )
            fig.add_trace(
                go.Scatterpolar(
                    r=np.concatenate([team_norm, [team_norm[0]]]),
                    theta=labels + [labels[0]],
                    fill="toself",
                    name=f"{TEAM_LABEL.get(team, 'Team')} avg",
                    line=dict(color="#71717a", width=2, dash="dot"),
                    fillcolor="rgba(113,113,122,0.12)",
                )
            )
            fig.update_polars(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1.05], showticklabels=False, gridcolor="#27272a"),
                angularaxis=dict(gridcolor="#27272a", linecolor="#27272a"),
            )
            styled_plot(
                fig,
                height=380,
                margin=dict(l=40, r=40, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15),
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------


with tab_heatmaps:
    if positions_df.empty:
        st.warning("positions.jsonl is missing or empty.")
    else:
        # Detect calibration that maps points outside the board. Calibration
        # noise of a few pixels is normal; gross overflow (> 20%) usually
        # means TL/BL/TR/BR were misplaced. Either way, the heatmap still
        # renders because we clamp to the board's visible extent.
        oob_count = int(
            ((positions_df["x"] < 0) | (positions_df["x"] > DEFAULT_BOARD_W)
             | (positions_df["y"] < 0) | (positions_df["y"] > DEFAULT_BOARD_H)).sum()
        )
        if oob_count > 0:
            oob_pct = 100.0 * oob_count / len(positions_df)
            st.caption(
                f"{oob_count:,} of {len(positions_df):,} positions ({oob_pct:.0f}%) "
                f"lie outside the {DEFAULT_BOARD_W}\u00d7{DEFAULT_BOARD_H} pitch area "
                "\u2014 calibration drift. Clamped to the board edge so the heatmap renders."
            )

        mode = st.radio(
            "Group by", ["All players", "Per team", "Per track"], horizontal=True,
        )
        bg_url = pitch_background_png(DEFAULT_BOARD_W, DEFAULT_BOARD_H)

        def _clamp(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            out["x"] = out["x"].clip(0, DEFAULT_BOARD_W)
            out["y"] = out["y"].clip(0, DEFAULT_BOARD_H)
            return out

        def _heatmap(df: pd.DataFrame, title: str) -> go.Figure:
            df = _clamp(df)
            fig = go.Figure(
                go.Histogram2dContour(
                    x=df["x"],
                    y=df["y"],
                    colorscale="Hot",
                    opacity=0.6,
                    ncontours=18,
                    showscale=False,
                )
            )
            fig.add_layout_image(
                dict(
                    source=bg_url,
                    xref="x", yref="y",
                    x=0, y=0,
                    sizex=DEFAULT_BOARD_W, sizey=DEFAULT_BOARD_H,
                    sizing="stretch",
                    opacity=1.0,
                    layer="below",
                )
            )
            fig.update_xaxes(range=[0, DEFAULT_BOARD_W], showgrid=False, zeroline=False, visible=False)
            fig.update_yaxes(
                range=[DEFAULT_BOARD_H, 0],
                showgrid=False, zeroline=False, visible=False,
                scaleanchor="x", scaleratio=DEFAULT_BOARD_H / DEFAULT_BOARD_W,
            )
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text=title, font=dict(size=14, color="#e7ece8")),
                height=360,
            )
            return fig

        if mode == "All players":
            st.plotly_chart(
                _heatmap(positions_df, "All on-pitch positions"),
                use_container_width=True,
            )
        elif mode == "Per team":
            cols = st.columns(2, gap="medium")
            for i, team_id in enumerate([0, 1]):
                team_df = positions_df[positions_df["team"] == team_id]
                with cols[i]:
                    if team_df.empty:
                        st.info(f"No data for {TEAM_LABEL[team_id]}.")
                    else:
                        st.plotly_chart(_heatmap(team_df, TEAM_LABEL[team_id]), use_container_width=True)
        else:
            st.caption(
                "Per-track heatmaps reflect one continuous detection window. "
                "A real player may have several tracks across the match."
            )
            player_ids = sorted(positions_df["id"].unique().tolist())
            selected = st.selectbox("Track", player_ids, format_func=lambda i: f"#{int(i)}")
            player_df = positions_df[positions_df["id"] == selected]
            st.plotly_chart(_heatmap(player_df, f"Track #{int(selected)}"), use_container_width=True)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


with tab_replay:
    if not positions:
        st.warning("positions.jsonl is missing or empty.")
    else:
        n_frames = len(positions)
        st.session_state.setdefault("replay_idx", 0)
        st.session_state.setdefault("replay_playing", False)

        per_frame_count = pd.DataFrame(
            {
                "frame": [r["frame"] for r in positions],
                "tracks": [len(r.get("players", [])) for r in positions],
            }
        )
        fig = px.area(
            per_frame_count, x="frame", y="tracks",
            labels={"frame": "", "tracks": "Tracks visible"},
        )
        fig.update_traces(line_color=ACCENT, fillcolor="rgba(216,92,125,0.15)")
        fig.add_vline(
            x=int(st.session_state["replay_idx"]),
            line_color=ACCENT, line_width=2, line_dash="dot",
        )
        styled_plot(fig, height=120, margin=dict(l=20, r=20, t=10, b=20))
        fig.update_yaxes(visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Estimate the source video's effective fps from the JSONL timestamps.
        # Falls back to 25 if we can't infer it (e.g. fewer than 2 frames).
        if n_frames >= 2:
            inferred_fps = (n_frames - 1) / max(positions[-1]["t"] - positions[0]["t"], 1e-6)
            source_fps = float(np.clip(inferred_fps, 1.0, 60.0))
        else:
            source_fps = 25.0

        col_c1, col_c2, col_speed, col_slider = st.columns([1, 1, 1, 5])
        with col_c1:
            if st.button(
                "Pause" if st.session_state["replay_playing"] else "Play",
                use_container_width=True,
            ):
                st.session_state["replay_playing"] = not st.session_state["replay_playing"]
        with col_c2:
            if st.button("Reset", use_container_width=True):
                st.session_state["replay_idx"] = 0
                st.session_state["replay_playing"] = False
        with col_speed:
            speed = st.select_slider(
                "Speed", options=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0], value=1.0,
                label_visibility="collapsed",
                help=f"Playback speed multiplier (source fps \u2248 {source_fps:.0f}).",
            )
        with col_slider:
            idx = st.slider(
                "Frame", min_value=0, max_value=n_frames - 1,
                value=int(st.session_state["replay_idx"]), step=1,
                label_visibility="collapsed",
            )
            st.session_state["replay_idx"] = idx

        record = positions[idx]
        players, ball = frame_to_players(record)

        # Sync mode toggle: side-by-side camera + tactical board, both jumping
        # to record["t"]. The camera video uses st.video(start_time=t) so the
        # browser seeks to that moment when the slider changes.
        camera_available = (
            camera_video_path.exists()
            and _classify_mp4(camera_video_path) == "h264"
        )
        sync_mode = False
        if camera_available:
            sync_mode = st.checkbox(
                "Sync camera video to the timeline",
                value=False,
                help=(
                    "Plays `camera.mp4` from the slider's current timestamp. "
                    "Requires an H.264-encoded camera video."
                ),
            )

        if sync_mode:
            col_cam, col_board, col_meta = st.columns([4, 3, 2], gap="medium")
            with col_cam:
                st.markdown("**Camera**")
                st.video(str(camera_video_path), start_time=float(record["t"]))
                st.caption(
                    f"Seeked to **t = {record['t']:.2f}s**. Scrub the frame "
                    "slider above to re-seek."
                )
        else:
            col_board, col_meta = st.columns([3, 1], gap="medium")

        with col_board:
            st.markdown("**Tactical board**")
            board_drawer = TacticalBoard(DEFAULT_BOARD_W, DEFAULT_BOARD_H)
            img = board_drawer.draw_state(players, ball)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)
            st.caption(
                f"Frame **{record['frame']}** &middot; t = **{record['t']:.2f}s** "
                f"&middot; {len(players)} track(s) &middot; "
                f"ball: {'on board' if ball is not None else 'not visible'}"
            )

        with col_meta:
            st.markdown("**On the pitch**")
            if players:
                tbl = pd.DataFrame(
                    [
                        {
                            "Track": pid,
                            "Team": TEAM_LABEL.get(team, "?"),
                            "x": round(float(pos[0]), 1),
                            "y": round(float(pos[1]), 1),
                        }
                        for pid, pos, team in players
                    ]
                )
                st.dataframe(tbl, use_container_width=True, height=320, hide_index=True)
            else:
                st.caption("_No tracks visible._")

        if st.session_state["replay_playing"] and idx < n_frames - 1:
            st.session_state["replay_idx"] = idx + 1
            # Sleep for one source frame, scaled by the playback speed.
            time.sleep(max(1.0 / (source_fps * speed), 0.01))
            st.rerun()


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


with tab_video:
    any_video = False
    if board_video_path.exists() and camera_video_path.exists():
        any_video = True
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            _render_video_block("Tactical board", board_video_path)
        with c2:
            _render_video_block("Annotated camera", camera_video_path)
    elif board_video_path.exists():
        any_video = True
        _render_video_block("Tactical board", board_video_path)
    elif camera_video_path.exists():
        any_video = True
        _render_video_block("Annotated camera", camera_video_path)

    if not any_video:
        st.info(
            "No video files in this run. Use the **Analyse** page with 'Record annotated "
            "camera' and 'Record tactical board' checked, or pass `--save-board-video` / "
            "`--save-video` to the CLI."
        )
