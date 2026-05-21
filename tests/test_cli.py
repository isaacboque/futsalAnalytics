"""Tests for the argparse CLI wiring (no actual pipeline execution)."""

from pathlib import Path

import pytest

from futsal_analytics.__main__ import build_parser


def test_defaults():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.url is None
    assert args.start == "0"
    assert args.calibration is None
    assert args.no_gui is False
    assert args.device == "auto"
    assert args.log_level == "INFO"
    assert args.max_frames == 0
    assert args.retrain_every == 600
    assert args.skip_when_behind == 0.0


def test_url_and_start():
    parser = build_parser()
    args = parser.parse_args(["--url", "https://youtube.com/watch?v=X", "--start", "01:30"])
    assert args.url == "https://youtube.com/watch?v=X"
    assert args.start == "01:30"


def test_output_paths_become_path_objects():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--save-video", "out.mp4",
            "--save-positions", "pos.jsonl",
            "--save-kpis", "kpis.csv",
            "--save-calibration", "cal.npy",
        ]
    )
    assert isinstance(args.save_video, Path)
    assert isinstance(args.save_positions, Path)
    assert isinstance(args.save_kpis, Path)
    assert isinstance(args.save_calibration, Path)


def test_calibration_load_path():
    parser = build_parser()
    args = parser.parse_args(["--calibration", "cal.npy"])
    assert args.calibration == Path("cal.npy")


def test_device_choices():
    parser = build_parser()
    for d in ("auto", "cpu", "cuda"):
        args = parser.parse_args(["--device", d])
        assert args.device == d


def test_invalid_device_rejected():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--device", "tpu"])


def test_log_level_choices():
    parser = build_parser()
    for lvl in ("DEBUG", "INFO", "WARNING", "ERROR"):
        args = parser.parse_args(["--log-level", lvl])
        assert args.log_level == lvl


def test_no_gui_flag():
    parser = build_parser()
    args = parser.parse_args(["--no-gui", "--calibration", "cal.npy"])
    assert args.no_gui is True


def test_max_frames_and_retrain():
    parser = build_parser()
    args = parser.parse_args(["--max-frames", "100", "--retrain-every", "500"])
    assert args.max_frames == 100
    assert args.retrain_every == 500


def test_skip_when_behind_float():
    parser = build_parser()
    args = parser.parse_args(["--skip-when-behind", "2.5"])
    assert args.skip_when_behind == 2.5


def test_imgsz_default_and_override():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.imgsz == 640
    args = parser.parse_args(["--imgsz", "320"])
    assert args.imgsz == 320


def test_frame_stride_default_and_override():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.frame_stride == 1
    args = parser.parse_args(["--frame-stride", "3"])
    assert args.frame_stride == 3
