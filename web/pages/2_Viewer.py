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
    TEAM_COLOR_HEX,
    TEAM_LABEL,
    file_mtime,
    frame_to_players,
    inject_css,
    load_kpis,
    load_positions,
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

positions, n_bad_positions = load_positions(str(positions_path), file_mtime(positions_path))
kpis_df_raw = load_kpis(str(kpis_path), file_mtime(kpis_path))
positions_df_raw = positions_to_dataframe(positions)

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

tab_overview, tab_players, tab_heatmaps, tab_replay, tab_video = st.tabs(
    ["Overview", "Tracks", "Heatmaps", "Replay", "Video"]
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
        leaderboard["label"] = leaderboard["track_id"].apply(lambda i: f"#{int(i)}")
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
        st.info(
            "These rows are **per-track**, not per-player. The same physical player "
            "may appear in several rows. Adjust the sidebar filter to hide short "
            "fragments.",
            icon=":material/info:",
        )

        display_df = kpis_df.copy()
        display_df["Team"] = display_df["team"].map(TEAM_LABEL).fillna("Unknown")
        display_df = display_df[
            ["track_id", "Team", "distance_m", "top_speed_ms",
             "sprint_count", "possession_s", "duel_s", "seen_s"]
        ].sort_values("seen_s", ascending=False).reset_index(drop=True)

        st.markdown("##### All tracks (filtered)")
        st.dataframe(
            display_df,
            use_container_width=True,
            height=320,
            hide_index=True,
            column_config={
                "track_id": st.column_config.NumberColumn("Track", width="small"),
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
            player_ids = sorted(kpis_df["track_id"].unique().tolist())
            selected_id = st.selectbox(
                "Track",
                player_ids,
                format_func=lambda i: f"#{int(i)}",
            )
            player_row = kpis_df[kpis_df["track_id"] == selected_id].iloc[0]
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

            squad_max = kpis_df[metric_cols].max().values.astype(float)
            squad_max[squad_max == 0] = 1.0
            player_norm = player_vals / squad_max
            team_norm = team_avg / squad_max

            fig = go.Figure()
            fig.add_trace(
                go.Scatterpolar(
                    r=np.concatenate([player_norm, [player_norm[0]]]),
                    theta=labels + [labels[0]],
                    fill="toself",
                    name=f"Track #{int(selected_id)}",
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
        mode = st.radio("Group by", ["Per team", "Per track"], horizontal=True)
        bg_url = pitch_background_png(DEFAULT_BOARD_W, DEFAULT_BOARD_H)

        def _heatmap(df: pd.DataFrame, title: str) -> go.Figure:
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

        if mode == "Per team":
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

        col_c1, col_c2, col_slider = st.columns([1, 1, 6])
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
        with col_slider:
            idx = st.slider(
                "Frame", min_value=0, max_value=n_frames - 1,
                value=int(st.session_state["replay_idx"]), step=1,
                label_visibility="collapsed",
            )
            st.session_state["replay_idx"] = idx

        record = positions[idx]
        players, ball = frame_to_players(record)

        col_board, col_meta = st.columns([3, 1], gap="medium")
        with col_board:
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
            time.sleep(0.05)
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
            st.markdown("##### Tactical board")
            st.video(str(board_video_path))
        with c2:
            st.markdown("##### Annotated camera")
            st.video(str(camera_video_path))
    elif board_video_path.exists():
        any_video = True
        st.markdown("##### Tactical board")
        st.video(str(board_video_path))
    elif camera_video_path.exists():
        any_video = True
        st.markdown("##### Annotated camera")
        st.video(str(camera_video_path))

    if not any_video:
        st.info(
            "No video files in this run. Use the **Analyse** page with 'Record annotated "
            "camera' and 'Record tactical board' checked, or pass `--save-board-video` / "
            "`--save-video` to the CLI."
        )
