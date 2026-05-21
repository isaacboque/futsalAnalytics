"""
Roster page — assign track IDs to real players.

ByteTrack assigns a fresh ID whenever a player is occluded, leaves the
frame, or is re-acquired after lighting changes, so one physical player
typically appears under several track IDs across a match. This page lets
you define each team's roster once and drag tracks onto roster slots, so
per-player KPIs aggregate across all the fragments of the same player.

The result is saved to ``<output-dir>/roster.json`` and picked up
automatically by the Viewer page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _shared import (
    ROSTER_FILENAME,
    TEAM_LABEL,
    all_players,
    apply_roster_to_kpis,
    file_mtime,
    inject_css,
    load_kpis,
    load_positions,
    load_roster,
    output_dir_picker,
    render_sidebar_brand,
    save_roster,
    suggest_track_merges,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------


st.set_page_config(
    page_title="Roster · Futsal Analytics",
    page_icon=":soccer:",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
render_sidebar_brand()

out_dir = output_dir_picker(default="out")
kpis_path = out_dir / "kpis.csv"
positions_path = out_dir / "positions.jsonl"
roster_path = out_dir / ROSTER_FILENAME

kpis_df = load_kpis(str(kpis_path), file_mtime(kpis_path))
positions, _ = load_positions(str(positions_path), file_mtime(positions_path))
roster = load_roster(roster_path) or {
    "teams": {"0": {"label": TEAM_LABEL[0], "players": []},
              "1": {"label": TEAM_LABEL[1], "players": []}},
    "assignments": {},
}


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


st.markdown(
    '<div class="fa-hero"><h1>Roster</h1>'
    '<span class="tag">track \u2192 player</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="fa-meta">Assign each tracked detection fragment to a real '
    'player so per-player KPIs aggregate correctly.</p>',
    unsafe_allow_html=True,
)

if kpis_df is None or kpis_df.empty:
    st.warning(
        "`kpis.csv` not found in the selected output directory. Run an analysis "
        "from the **Analyse** page first."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Step 1 — Define rosters
# ---------------------------------------------------------------------------


st.markdown("### 1\u2002\u00b7\u2002Define team rosters")
st.caption(
    "One player per line, optionally with a jersey number. "
    "Examples: `Iniesta #6`, `Player A`, `Jordi 7`."
)


def _parse_roster_text(text: str, team_str: str) -> list[dict]:
    out = []
    for i, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line:
            continue
        number = None
        name = line
        # Pull out a trailing `#N` or ` N` jersey number
        for sep in ("#", " "):
            if sep in line:
                left, _, right = line.rpartition(sep)
                if right.strip().isdigit() and left.strip():
                    name = left.strip()
                    number = int(right.strip())
                    break
        out.append({
            "id": f"t{team_str}_{i}_{name.replace(' ', '_').lower()}",
            "name": name,
            "number": number,
        })
    return out


def _format_roster(players: list[dict]) -> str:
    lines = []
    for p in players:
        num = p.get("number")
        lines.append(f"{p.get('name', '?')}" + (f" #{num}" if num else ""))
    return "\n".join(lines)


col_a, col_b = st.columns(2, gap="large")
for col, team_str in [(col_a, "0"), (col_b, "1")]:
    with col:
        team_data = roster["teams"].setdefault(
            team_str,
            {"label": TEAM_LABEL[int(team_str)], "players": []},
        )
        team_data["label"] = st.text_input(
            f"{TEAM_LABEL[int(team_str)]} label",
            value=team_data.get("label", TEAM_LABEL[int(team_str)]),
            key=f"roster_label_{team_str}",
        )
        text = st.text_area(
            f"{team_data['label']} players",
            value=_format_roster(team_data.get("players", [])),
            height=220,
            key=f"roster_text_{team_str}",
        )
        team_data["players"] = _parse_roster_text(text, team_str)
        st.caption(f"{len(team_data['players'])} player(s) parsed.")


# ---------------------------------------------------------------------------
# Step 2 — Assign tracks to players
# ---------------------------------------------------------------------------


st.markdown("### 2\u2002\u00b7\u2002Assign tracks to players")

# Build a single sorted track table, with each track's current assignment.
players_flat = all_players(roster)
player_choices = ["(unassigned)"] + [pid for pid, _, _ in players_flat]
player_labels = {"(unassigned)": "(unassigned)"}
for pid, label, _ in players_flat:
    player_labels[pid] = label

if not players_flat:
    st.info(
        "Add at least one player to a roster above before assigning tracks.",
        icon=":material/info:",
    )
else:
    assignments = roster.get("assignments", {})
    track_df = kpis_df.copy()
    track_df["Team"] = track_df["team"].map(TEAM_LABEL).fillna("Unknown")
    track_df["Assigned to"] = track_df["track_id"].astype(str).map(
        lambda t: assignments.get(t, "(unassigned)")
    )
    track_df = track_df.sort_values("seen_s", ascending=False).reset_index(drop=True)

    st.caption(
        f"{len(track_df)} tracks total. Heaviest first \u2014 assign the long ones, "
        "ignore the short fragments."
    )
    min_seen = st.slider(
        "Hide tracks shorter than (s)",
        min_value=0.0, max_value=float(track_df["seen_s"].max() or 1.0),
        value=2.0, step=0.5,
    )
    visible = track_df[track_df["seen_s"] >= min_seen].copy()

    edited = st.data_editor(
        visible[["track_id", "Team", "seen_s", "distance_m",
                 "top_speed_ms", "Assigned to"]],
        hide_index=True,
        use_container_width=True,
        height=420,
        disabled=["track_id", "Team", "seen_s", "distance_m", "top_speed_ms"],
        column_config={
            "track_id": st.column_config.NumberColumn("Track"),
            "Team": st.column_config.TextColumn("Team", width="small"),
            "seen_s": st.column_config.NumberColumn("Seen (s)", format="%.1f"),
            "distance_m": st.column_config.NumberColumn("Distance (m)", format="%.1f"),
            "top_speed_ms": st.column_config.NumberColumn("Top (m/s)", format="%.2f"),
            "Assigned to": st.column_config.SelectboxColumn(
                "Assigned to",
                options=player_choices,
                required=True,
            ),
        },
        key="roster_editor",
    )

    # Commit edits back into the working roster dict before saving.
    if edited is not None:
        for _, row in edited.iterrows():
            tid = str(int(row["track_id"]))
            pid = row["Assigned to"]
            if pid == "(unassigned)":
                roster["assignments"].pop(tid, None)
            else:
                roster["assignments"][tid] = pid

    # Visible at-a-glance counts
    counts_cols = st.columns(3)
    n_assigned = sum(1 for v in roster["assignments"].values() if v != "(unassigned)")
    counts_cols[0].metric("Tracks total", len(track_df))
    counts_cols[1].metric("Assigned", n_assigned)
    counts_cols[2].metric("Players defined", len(players_flat))


# ---------------------------------------------------------------------------
# Step 2.5 — Auto-suggest track merges
# ---------------------------------------------------------------------------


if positions and players_flat:
    st.markdown("### 2.5\u2002\u00b7\u2002Suggested merges (auto)")
    st.caption(
        "Heuristic: track A ends, track B starts within a few seconds nearby "
        "and on the same team \u2014 probably the same player. Pick a target "
        "player below and accept the suggestions you trust; each accepted pair "
        "assigns *both* track IDs to that player."
    )

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        gap_s = st.slider("Max time gap (s)", 0.5, 8.0, 3.0, 0.5,
                          help="How long after track A ends can track B start?")
    with col_g2:
        gap_m = st.slider("Max metric distance (m)", 1.0, 20.0, 8.0, 1.0,
                          help="How far can the player have moved during the gap?")

    suggestions = suggest_track_merges(
        positions, max_gap_seconds=gap_s, max_distance_m=gap_m,
    )

    if not suggestions:
        st.caption("_No merge candidates found within these thresholds._")
    else:
        st.caption(
            f"Found {len(suggestions)} candidate merge(s). Showing the top "
            "20 (best matches first). Pick the player to assign both tracks "
            "to, then click Accept."
        )
        for a, b, score in suggestions[:20]:
            cols = st.columns([2, 2, 2, 4, 2])
            current_a = roster["assignments"].get(str(a), "(unassigned)")
            current_b = roster["assignments"].get(str(b), "(unassigned)")
            cols[0].markdown(f"**#{a}** \u2192 #{b}")
            cols[1].caption(f"score {score:.2f}")
            cols[2].caption(
                f"now: {('—' if current_a == '(unassigned)' else current_a)} / "
                f"{('—' if current_b == '(unassigned)' else current_b)}"
            )
            target = cols[3].selectbox(
                "Assign both to",
                ["(skip)"] + [pid for pid, _, _ in players_flat],
                index=0,
                key=f"merge_target_{a}_{b}",
                label_visibility="collapsed",
                format_func=lambda pid: (
                    "(skip)" if pid == "(skip)" else
                    next((label for pid_, label, _ in players_flat
                          if pid_ == pid), pid)
                ),
            )
            if cols[4].button("Accept", key=f"merge_accept_{a}_{b}",
                              disabled=(target == "(skip)"),
                              use_container_width=True):
                roster["assignments"][str(a)] = target
                roster["assignments"][str(b)] = target
                save_roster(roster_path, roster)
                st.toast(f"Assigned #{a} and #{b} to {target}",
                         icon=":material/merge:")
                st.cache_data.clear()
                st.rerun()


# ---------------------------------------------------------------------------
# Step 3 — Save + preview aggregation
# ---------------------------------------------------------------------------


st.markdown("### 3\u2002\u00b7\u2002Save \u0026 preview")

save_col, _ = st.columns([1, 4])
with save_col:
    if st.button("Save roster", type="primary", use_container_width=True):
        save_roster(roster_path, roster)
        st.toast(f"Saved {roster_path.name}", icon=":material/save:")
        # Force the Viewer's cached load to pick up the new mtime.
        st.cache_data.clear()

if roster.get("assignments"):
    st.markdown("##### Aggregated KPIs (preview)")
    agg = apply_roster_to_kpis(kpis_df, roster)
    # Move display column first
    if "display" in agg.columns:
        cols = ["display"] + [c for c in agg.columns if c != "display"]
        agg = agg[cols]
    agg["Team"] = agg["team"].map(TEAM_LABEL).fillna("Unknown")
    agg = agg[[
        "display", "Team", "track_count", "seen_s",
        "distance_m", "top_speed_ms", "sprint_count",
        "possession_s", "duel_s",
    ]].sort_values("distance_m", ascending=False).reset_index(drop=True)

    st.dataframe(
        agg,
        use_container_width=True,
        hide_index=True,
        height=320,
        column_config={
            "display": st.column_config.TextColumn("Player"),
            "Team": st.column_config.TextColumn("Team", width="small"),
            "track_count": st.column_config.NumberColumn("Tracks"),
            "seen_s": st.column_config.NumberColumn("Seen (s)", format="%.1f"),
            "distance_m": st.column_config.NumberColumn("Distance (m)", format="%.1f"),
            "top_speed_ms": st.column_config.NumberColumn("Top (m/s)", format="%.2f"),
            "sprint_count": st.column_config.NumberColumn("Sprints"),
            "possession_s": st.column_config.NumberColumn("Possession (s)", format="%.1f"),
            "duel_s": st.column_config.NumberColumn("Duels (s)", format="%.1f"),
        },
    )
    st.caption(
        "Switch to the **Viewer** page and toggle 'Aggregate by Player' in "
        "the sidebar to see these numbers in the dashboards."
    )
