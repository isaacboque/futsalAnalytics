"""Tests for web/_shared.py — the robust position/kpi loaders."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")  # web extras are optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web"))

from _shared import load_kpis, load_positions


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
