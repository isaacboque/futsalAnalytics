"""Tests for web/_shared.py — the robust position/kpi loaders."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")  # web extras are optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web"))

import pandas as pd
from _shared import (
    apply_roster_to_kpis,
    load_kpis,
    load_positions,
    load_roster,
    player_label,
    save_roster,
)


class TestLoadPositionsRobust:
    def test_empty_file(self, tmp_path):
        p = tmp_path / "positions.jsonl"
        p.write_text("", encoding="utf-8")
        records, skipped = load_positions(str(p), p.stat().st_mtime)
        assert records == []
        assert skipped == 0

    def test_missing_file(self, tmp_path):
        records, skipped = load_positions(str(tmp_path / "nope.jsonl"), 0)
        assert records == []
        assert skipped == 0

    def test_valid_lines_only(self, tmp_path):
        p = tmp_path / "positions.jsonl"
        lines = [
            {"frame": 0, "t": 0.0, "players": [], "ball": None},
            {"frame": 1, "t": 0.04, "players": [{"id": 1, "team": 0, "x": 1, "y": 2}], "ball": None},
        ]
        p.write_text("\n".join(json.dumps(l) for l in lines), encoding="utf-8")
        records, skipped = load_positions(str(p), p.stat().st_mtime)
        assert len(records) == 2
        assert skipped == 0

    def test_skips_garbage_lines(self, tmp_path):
        p = tmp_path / "positions.jsonl"
        good = {"frame": 0, "t": 0.0, "players": [], "ball": None}
        text = (
            json.dumps(good) + "\n"
            + "this is not json\n"
            + "{\"truncated\":\n"
            + json.dumps(good) + "\n"
            + "  \n"     # whitespace-only — silently skipped
        )
        p.write_text(text, encoding="utf-8")
        records, skipped = load_positions(str(p), p.stat().st_mtime)
        assert len(records) == 2     # only the two valid records survive
        assert skipped == 2          # two non-empty malformed lines

    def test_handles_invalid_utf8(self, tmp_path):
        p = tmp_path / "positions.jsonl"
        good = json.dumps({"frame": 0, "t": 0.0, "players": [], "ball": None})
        # Mix valid line + line with bad bytes
        p.write_bytes(good.encode("utf-8") + b"\n" + b"\xff\xfe\xff{\"oops\"\n" + good.encode("utf-8"))
        records, skipped = load_positions(str(p), p.stat().st_mtime)
        # Both good lines survive; the garbage one is counted as skipped
        assert len(records) == 2
        assert skipped == 1


class TestLoadKpis:
    def test_missing_returns_none(self, tmp_path):
        assert load_kpis(str(tmp_path / "nope.csv"), 0) is None

    def test_empty_returns_none(self, tmp_path):
        p = tmp_path / "kpis.csv"
        p.write_text("", encoding="utf-8")
        assert load_kpis(str(p), p.stat().st_mtime) is None

    def test_normal(self, tmp_path):
        p = tmp_path / "kpis.csv"
        p.write_text("track_id,team,distance_m\n1,0,42.5\n2,1,17.0\n", encoding="utf-8")
        df = load_kpis(str(p), p.stat().st_mtime)
        assert df is not None
        assert list(df.columns) == ["track_id", "team", "distance_m"]
        assert len(df) == 2

    def test_skips_bad_rows(self, tmp_path):
        p = tmp_path / "kpis.csv"
        # Header + valid row + row with wrong column count
        p.write_text(
            "track_id,team,distance_m\n"
            "1,0,42.5\n"
            "broken,row,with,too,many,fields\n"
            "2,1,17.0\n",
            encoding="utf-8",
        )
        df = load_kpis(str(p), p.stat().st_mtime)
        assert df is not None
        # The on_bad_lines='skip' option should keep us going
        assert len(df) >= 2


class TestRoster:
    def _sample_roster(self):
        return {
            "teams": {
                "0": {"label": "Home", "players": [
                    {"id": "h_1", "name": "Iniesta", "number": 6},
                    {"id": "h_2", "name": "Jordi", "number": 10},
                ]},
                "1": {"label": "Away", "players": [
                    {"id": "a_1", "name": "Ricardinho", "number": 11},
                ]},
            },
            "assignments": {
                "1": "h_1",
                "2": "h_1",   # two tracks rolled up into Iniesta
                "3": "a_1",
            },
        }

    def test_save_load_roundtrip(self, tmp_path):
        path = tmp_path / "roster.json"
        original = self._sample_roster()
        save_roster(path, original)
        loaded = load_roster(path)
        assert loaded["teams"]["0"]["label"] == "Home"
        assert loaded["assignments"]["2"] == "h_1"

    def test_save_is_atomic(self, tmp_path):
        path = tmp_path / "roster.json"
        save_roster(path, self._sample_roster())
        assert path.exists()
        assert not (tmp_path / "roster.tmp.json").exists()

    def test_load_missing_returns_none(self, tmp_path):
        assert load_roster(tmp_path / "nope.json") is None

    def test_load_garbage_returns_none(self, tmp_path):
        p = tmp_path / "roster.json"
        p.write_text("this is not json", encoding="utf-8")
        assert load_roster(p) is None

    def test_player_label(self):
        roster = self._sample_roster()
        assert player_label(roster, "h_1") == "Iniesta #6"
        assert player_label(roster, "a_1") == "Ricardinho #11"
        # Falls back to the id when the player isn't in the roster
        assert player_label(roster, "ghost") == "ghost"

    def test_apply_roster_to_kpis_aggregates(self):
        kpis = pd.DataFrame([
            # Two tracks both belong to Iniesta — should combine
            {"track_id": 1, "team": 0, "distance_m": 100.0,
             "top_speed_ms": 7.5, "sprint_count": 5, "possession_s": 10.0,
             "duel_s": 2.0, "seen_s": 30.0},
            {"track_id": 2, "team": 0, "distance_m": 200.0,
             "top_speed_ms": 9.0, "sprint_count": 10, "possession_s": 5.0,
             "duel_s": 1.0, "seen_s": 60.0},
            # One track for Ricardinho
            {"track_id": 3, "team": 1, "distance_m": 150.0,
             "top_speed_ms": 8.0, "sprint_count": 7, "possession_s": 8.0,
             "duel_s": 3.0, "seen_s": 50.0},
            # One unassigned track — goes to "_unassigned"
            {"track_id": 99, "team": 1, "distance_m": 5.0,
             "top_speed_ms": 4.0, "sprint_count": 0, "possession_s": 0.0,
             "duel_s": 0.0, "seen_s": 2.0},
        ])
        roster = self._sample_roster()
        agg = apply_roster_to_kpis(kpis, roster)

        iniesta = agg[agg["player_id"] == "h_1"].iloc[0]
        assert iniesta["distance_m"] == 300.0
        assert iniesta["sprint_count"] == 15
        assert iniesta["top_speed_ms"] == 9.0    # max, not sum
        assert iniesta["track_count"] == 2
        assert iniesta["display"] == "Iniesta #6"

        unassigned = agg[agg["player_id"] == "_unassigned"].iloc[0]
        assert unassigned["track_count"] == 1
        assert unassigned["display"] == "Unassigned tracks"

    def test_apply_roster_no_assignments_groups_all_as_unassigned(self):
        kpis = pd.DataFrame([
            {"track_id": 1, "team": 0, "distance_m": 50.0,
             "top_speed_ms": 6.0, "sprint_count": 2, "possession_s": 0.0,
             "duel_s": 0.0, "seen_s": 10.0},
        ])
        agg = apply_roster_to_kpis(kpis, None)
        assert len(agg) == 1
        assert agg.iloc[0]["player_id"] == "_unassigned"
