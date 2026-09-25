"""PivotIQ command-line interface.

Entry point: ``pivotiq analyze <video.mp4> [options]``

Milestone 1 — CLI skeleton only.  The analysis pipeline is stubbed; it will
be wired in milestones 2–5.  The command is runnable and all flags are parsed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="pivotiq",
    help="PivotIQ — Automated futsal moment-intelligence tool.",
    add_completion=False,
    no_args_is_help=True,
    invoke_without_command=False,
)

logger = logging.getLogger(__name__)


@app.command()
def analyze(
    video: Path = typer.Argument(..., help="Input MP4 video file."),
    ball_weights: Optional[Path] = typer.Option(
        None, "--ball-weights", help="Fine-tuned ball detector weights (.pt)."
    ),
    keypoint_weights: Optional[Path] = typer.Option(
        None, "--keypoint-weights", help="Pitch keypoint detector weights (.pt)."
    ),
    out_video: Optional[Path] = typer.Option(
        None, "--out-video", help="Output annotated MP4 path."
    ),
    report_dir: Optional[Path] = typer.Option(
        None, "--report-dir", help="Directory for report outputs (PNG/JSON/clips)."
    ),
    device: str = typer.Option(
        "auto", "--device", help="Compute device: auto | cpu | cuda | mps."
    ),
    max_frames: Optional[int] = typer.Option(
        None, "--max-frames", help="Limit processing to N frames (useful for smoke tests)."
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", help="Path to TOML config file (overrides built-in defaults)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    """Analyse a futsal match video and produce an annotated video and/or a report.

    When no ``--ball-weights`` are given the pipeline falls back to a
    pretrained YOLO model (less accurate for small futsal balls).
    When no ``--keypoint-weights`` are given a MockRegistration (identity
    homography) is used — pixel coordinates are treated as pitch coordinates.
    """
    _setup_logging(verbose)

    if not video.exists():
        typer.echo(f"Error: video file not found: {video}", err=True)
        raise typer.Exit(1)
    if video.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
        typer.echo(f"Warning: unexpected file extension '{video.suffix}'. Continuing.", err=True)

    from pivotiq.config import PivotIQConfig

    cfg = PivotIQConfig.from_toml(config_file) if config_file else PivotIQConfig.default()
    if device != "auto":
        cfg.detection.device = device

    typer.echo("PivotIQ")
    typer.echo(f"  video           : {video}")
    typer.echo(f"  ball_weights    : {ball_weights or '(fallback YOLO)'}")
    typer.echo(f"  keypoint_weights: {keypoint_weights or '(MockRegistration)'}")
    typer.echo(f"  out_video       : {out_video or '(not requested)'}")
    typer.echo(f"  report_dir      : {report_dir or '(not requested)'}")
    typer.echo(f"  device          : {cfg.detection.device}")
    typer.echo(f"  max_frames      : {max_frames if max_frames is not None else 'all'}")
    typer.echo("")
    typer.echo(
        "Pipeline: milestone 2+ required for full analysis.  "
        "Run 'pivotiq analyze' again after completing the next milestone."
    )
    logger.info("Stub pipeline completed — milestone 2 will add real video processing.")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


@app.command()
def version() -> None:
    """Show PivotIQ version."""
    from pivotiq import __version__

    typer.echo(f"pivotiq {__version__}")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
