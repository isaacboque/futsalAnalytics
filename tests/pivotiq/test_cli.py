"""Unit tests for pivotiq.cli — no GPU, no weights, no real video required."""

import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pivotiq.cli import app

runner = CliRunner()


class TestHelp:
    def test_top_level_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "analyze" in result.output.lower()

    def test_analyze_help(self):
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        for flag in ("--ball-weights", "--keypoint-weights", "--out-video", "--report-dir"):
            assert flag in result.output


class TestAnalyzeCommand:
    def test_missing_video_arg(self):
        # No arguments → typer shows help (no_args_is_help=True on root app)
        result = runner.invoke(app, [])
        # Exit 0 with help text OR non-zero with usage error — either is fine
        assert "analyze" in result.output.lower() or result.exit_code != 0

    def test_nonexistent_video_exits_with_error(self):
        result = runner.invoke(app, ["analyze", "nonexistent_video.mp4"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "does not exist" in result.output.lower()

    def test_stub_run_with_real_file(self, tmp_path):
        """Creates a dummy .mp4 file and verifies the stub pipeline runs."""
        fake_mp4 = tmp_path / "match.mp4"
        fake_mp4.write_bytes(b"\x00" * 1024)  # not a real MP4, but file exists

        result = runner.invoke(
            app,
            [
                "analyze",
                str(fake_mp4),
                "--max-frames", "10",
                "--device", "cpu",
            ],
        )
        assert result.exit_code == 0
        assert "milestone" in result.output.lower()

    def test_stub_run_verbose(self, tmp_path):
        fake_mp4 = tmp_path / "match.mp4"
        fake_mp4.write_bytes(b"\x00" * 1024)
        result = runner.invoke(app, ["analyze", str(fake_mp4), "--verbose"])
        assert result.exit_code == 0

    def test_stub_run_with_report_dir(self, tmp_path):
        fake_mp4 = tmp_path / "match.mp4"
        fake_mp4.write_bytes(b"\x00" * 1024)
        result = runner.invoke(
            app,
            ["analyze", str(fake_mp4), "--report-dir", str(tmp_path / "report")],
        )
        assert result.exit_code == 0


class TestConfig:
    def test_toml_config_file(self, tmp_path):
        toml = tmp_path / "cfg.toml"
        toml.write_text(
            "[detection]\nplayer_confidence = 0.55\n[events]\nshot_speed_threshold_mps = 7.0\n",
            encoding="utf-8",
        )
        fake_mp4 = tmp_path / "match.mp4"
        fake_mp4.write_bytes(b"\x00" * 256)
        result = runner.invoke(
            app,
            ["analyze", str(fake_mp4), "--config", str(toml)],
        )
        assert result.exit_code == 0

    def test_nonexistent_config_raises(self, tmp_path):
        fake_mp4 = tmp_path / "match.mp4"
        fake_mp4.write_bytes(b"\x00" * 256)
        result = runner.invoke(
            app,
            ["analyze", str(fake_mp4), "--config", str(tmp_path / "missing.toml")],
        )
        assert result.exit_code != 0
