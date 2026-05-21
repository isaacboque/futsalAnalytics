"""Tests for KPITracker and PositionLogger."""

import json

import numpy as np
import pytest

from futsal_analytics.kpis import KPITracker, PositionLogger


# ---------------------------------------------------------------------------
# KPITracker
# ---------------------------------------------------------------------------


class TestKPITrackerBasics:
    def test_empty_returns_no_rows(self):
        kt = KPITracker(fps=25.0, board_width_px=700, board_height_px=350)
        assert kt.to_rows() == []

    def test_single_player_one_frame(self):
        kt = KPITracker(fps=25.0, board_width_px=700, board_height_px=350)
        kt.update(0, [(1, np.array([100.0, 100.0]), 0)], None)
        rows = kt.to_rows()
        assert len(rows) == 1
        assert rows[0]["track_id"] == 1
        assert rows[0]["team"] == 0
        # No movement yet, distance is 0
        assert rows[0]["distance_m"] == 0.0

    def test_skips_unassigned_track_ids(self):
        kt = KPITracker(fps=25.0, board_width_px=700, board_height_px=350)
        kt.update(0, [(-1, np.array([100.0, 100.0]), 0), (-2, np.array([200.0, 200.0]), 1)], None)
        # Both have negative IDs → both skipped
        assert kt.to_rows() == []


class TestKPITrackerDistance:
    def test_distance_accumulates_across_frames(self):
        # 5 px per frame at 25 fps = 0.286 m / 0.04s = 7.14 m/s (realistic walking/jogging)
        kt = KPITracker(fps=25.0, board_width_px=700, board_height_px=350)
        kt.update(0, [(1, np.array([100.0, 100.0]), 0)], None)
        kt.update(1, [(1, np.array([105.0, 100.0]), 0)], None)
        kt.update(2, [(1, np.array([110.0, 100.0]), 0)], None)
        rows = kt.to_rows()
        # Two steps of 5 px each = 10 px total → 10 * 40/700 = 0.57 m
        expected = 2 * 5 * 40 / 700
        assert rows[0]["distance_m"] == pytest.approx(expected, abs=0.1)

    def test_implausible_jump_is_ignored(self):
        kt = KPITracker(fps=25.0, board_width_px=700, board_height_px=350)
        # Teleport of 700 px in one frame at 25 fps → impossible speed
        kt.update(0, [(1, np.array([0.0, 0.0]), 0)], None)
        kt.update(1, [(1, np.array([700.0, 350.0]), 0)], None)
        rows = kt.to_rows()
        assert rows[0]["distance_m"] == 0.0


class TestKPITrackerSprints:
    def test_sprint_detected_above_threshold(self):
        # 5 m/s threshold; move 0.5 m in 0.04s (1 frame @ 25fps) = 12.5 m/s but capped, so adjust
        # Use 0.25 m per frame = 6.25 m/s → above threshold
        kt = KPITracker(fps=25.0, board_width_px=4000, board_height_px=2000)
        # m_per_px = 0.01, so 25 px = 0.25 m per frame = 6.25 m/s
        kt.update(0, [(1, np.array([0.0, 0.0]), 0)], None)
        kt.update(1, [(1, np.array([25.0, 0.0]), 0)], None)
        kt.update(2, [(1, np.array([50.0, 0.0]), 0)], None)
        rows = kt.to_rows()
        assert rows[0]["sprint_count"] >= 1


class TestKPITrackerPossession:
    def test_player_near_ball_gets_possession(self):
        kt = KPITracker(fps=25.0, board_width_px=700, board_height_px=350)
        # Ball at (100, 100), player at (101, 101) — very close
        kt.update(0, [(1, np.array([100.0, 100.0]), 0)], np.array([101.0, 101.0]))
        rows = kt.to_rows()
        assert rows[0]["possession_s"] > 0

    def test_player_far_from_ball_no_possession(self):
        kt = KPITracker(fps=25.0, board_width_px=700, board_height_px=350)
        kt.update(0, [(1, np.array([0.0, 0.0]), 0)], np.array([699.0, 349.0]))
        rows = kt.to_rows()
        assert rows[0]["possession_s"] == 0


class TestKPITrackerDuels:
    def test_opposing_team_proximity_logs_duel(self):
        kt = KPITracker(fps=25.0, board_width_px=700, board_height_px=350)
        # Two players from different teams, ~10 px = 0.4 m apart → within 1.5 m duel radius
        kt.update(0, [
            (1, np.array([100.0, 100.0]), 0),
            (2, np.array([105.0, 100.0]), 1),
        ], None)
        rows = kt.to_rows()
        # Both should have a duel frame logged
        assert all(r["duel_s"] > 0 for r in rows)

    def test_same_team_no_duel(self):
        kt = KPITracker(fps=25.0, board_width_px=700, board_height_px=350)
        kt.update(0, [
            (1, np.array([100.0, 100.0]), 0),
            (2, np.array([105.0, 100.0]), 0),
        ], None)
        rows = kt.to_rows()
        assert all(r["duel_s"] == 0 for r in rows)


class TestKPITrackerCSV:
    def test_save_csv_writes_header_and_rows(self, tmp_path):
        kt = KPITracker(fps=25.0, board_width_px=700, board_height_px=350)
        kt.update(0, [(1, np.array([100.0, 100.0]), 0)], None)
        kt.update(1, [(1, np.array([110.0, 100.0]), 0)], None)
        out = tmp_path / "kpis.csv"
        kt.save_csv(out)

        content = out.read_text()
        assert "track_id" in content
        assert "distance_m" in content

    def test_summary_returns_string(self):
        kt = KPITracker(fps=25.0, board_width_px=700, board_height_px=350)
        kt.update(0, [(1, np.array([0.0, 0.0]), 0)], None)
        s = kt.summary()
        assert "ID" in s
        assert "DIST" in s


# ---------------------------------------------------------------------------
# PositionLogger
# ---------------------------------------------------------------------------


class TestPositionLogger:
    def test_writes_one_jsonl_per_call(self, tmp_path):
        log_path = tmp_path / "positions.jsonl"
        pl = PositionLogger(log_path)
        pl.log(0, 0.0, [(1, np.array([100.0, 200.0]), 0)], np.array([50.0, 50.0]))
        pl.log(1, 0.04, [(1, np.array([105.0, 200.0]), 0)], None)
        pl.close()

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["frame"] == 0
        assert record["players"][0]["id"] == 1
        assert record["ball"] == {"x": 50.0, "y": 50.0}

        record1 = json.loads(lines[1])
        assert record1["ball"] is None

    def test_skips_players_without_track_id(self, tmp_path):
        log_path = tmp_path / "positions.jsonl"
        pl = PositionLogger(log_path)
        pl.log(0, 0.0, [(-1, np.array([100.0, 100.0]), 0)], None)
        pl.close()

        record = json.loads(log_path.read_text().strip())
        assert record["players"] == []
