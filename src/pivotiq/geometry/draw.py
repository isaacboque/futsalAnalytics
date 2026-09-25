"""Matplotlib utility for drawing a FIFA futsal pitch diagram.

Usage::

    from pivotiq.geometry.draw import draw_pitch
    fig, ax = draw_pitch()
    # overlay shot dots, player positions, etc. on *ax*
    fig.savefig("shot_map.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .pitch import (
    ARC_FOOT_Y1,
    ARC_FOOT_Y2,
    GOAL_DEPTH_M,
    GOAL_POST_Y1,
    GOAL_POST_Y2,
    HALF_LENGTH,
    HALF_WIDTH,
    KEYPOINTS,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    SUBSTITUTION_ZONE_HALF_LENGTH_M,
    center_circle_points,
    keeper_arc_points,
)


def draw_pitch(
    figsize: tuple[float, float] = (12.0, 6.0),
    pitch_color: str = "#3a7d2c",
    line_color: str = "white",
    linewidth: float = 1.5,
    ax: Optional[plt.Axes] = None,
) -> tuple[Figure, plt.Axes]:
    """Draw a blank FIFA futsal pitch.

    Parameters
    ----------
    figsize : (width, height) in inches, ignored when *ax* is supplied.
    pitch_color : background fill colour.
    line_color : colour of all pitch markings.
    linewidth : width of pitch lines.
    ax : existing Axes to draw into; if None, a new figure is created.

    Returns
    -------
    (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.set_facecolor(pitch_color)
    ax.set_xlim(-2.5, PITCH_LENGTH_M + 2.5)
    ax.set_ylim(PITCH_WIDTH_M + 2.5, -2.5)  # y increases downward
    ax.set_aspect("equal")
    ax.axis("off")

    lw = linewidth
    lc = line_color

    # --- Pitch outline ---
    ax.add_patch(
        mpatches.Rectangle(
            (0.0, 0.0),
            PITCH_LENGTH_M,
            PITCH_WIDTH_M,
            fill=False,
            edgecolor=lc,
            linewidth=lw,
        )
    )

    # --- Halfway line ---
    ax.plot([HALF_LENGTH, HALF_LENGTH], [0.0, PITCH_WIDTH_M], color=lc, linewidth=lw)

    # --- Centre circle + centre mark ---
    cc = center_circle_points()
    ax.plot(cc[:, 0], cc[:, 1], color=lc, linewidth=lw)
    ax.plot(*KEYPOINTS["center"], "o", color=lc, markersize=3)

    # --- Goals (rectangles extending outside the pitch) ---
    for goal_x, sign in [(0.0, -1), (PITCH_LENGTH_M, 1)]:
        ax.add_patch(
            mpatches.Rectangle(
                (goal_x + sign * GOAL_DEPTH_M, GOAL_POST_Y1),
                GOAL_DEPTH_M,
                GOAL_POST_Y2 - GOAL_POST_Y1,
                fill=False,
                edgecolor=lc,
                linewidth=lw,
            )
        )

    # --- Keeper arcs ---
    for side in ("left", "right"):
        arc = keeper_arc_points(side)
        ax.plot(arc[:, 0], arc[:, 1], color=lc, linewidth=lw)
        # Close arc to the goal-line feet with vertical lines
        if side == "left":
            ax.plot([0.0, 0.0], [ARC_FOOT_Y1, arc[0, 1]], color=lc, linewidth=lw)
            ax.plot([0.0, 0.0], [ARC_FOOT_Y2, arc[-1, 1]], color=lc, linewidth=lw)
        else:
            ax.plot(
                [PITCH_LENGTH_M, PITCH_LENGTH_M],
                [ARC_FOOT_Y2, arc[0, 1]],
                color=lc,
                linewidth=lw,
            )
            ax.plot(
                [PITCH_LENGTH_M, PITCH_LENGTH_M],
                [ARC_FOOT_Y1, arc[-1, 1]],
                color=lc,
                linewidth=lw,
            )

    # --- Penalty spots ---
    for name in (
        "left_penalty_6m",
        "right_penalty_6m",
        "left_penalty_10m",
        "right_penalty_10m",
    ):
        ax.plot(*KEYPOINTS[name], "o", color=lc, markersize=2.5)

    # --- Substitution zone ticks (bottom touchline) ---
    tick_h = 0.4
    for x_off in (
        -SUBSTITUTION_ZONE_HALF_LENGTH_M,
        SUBSTITUTION_ZONE_HALF_LENGTH_M,
    ):
        ax.plot(
            [HALF_LENGTH + x_off, HALF_LENGTH + x_off],
            [PITCH_WIDTH_M, PITCH_WIDTH_M + tick_h],
            color=lc,
            linewidth=lw,
        )

    assert fig is not None
    return fig, ax


def draw_shot_map(
    shot_pitchxy: Sequence[tuple[float, float]],
    outcomes: Sequence[str],
    figsize: tuple[float, float] = (12.0, 6.0),
) -> tuple[Figure, plt.Axes]:
    """Overlay shot dots on a pitch diagram.

    Parameters
    ----------
    shot_pitchxy : list of (x, y) pitch coordinates for each shot.
    outcomes : matching list of outcome strings ('goal', 'on_target', etc.).
    """
    fig, ax = draw_pitch(figsize=figsize)

    _colours = {
        "goal": "#ff4444",
        "on_target": "#ffaa00",
        "blocked": "#4488ff",
        "off_target": "#888888",
        "chance": "#aa44ff",
    }
    _markers = {
        "goal": "*",
        "on_target": "o",
        "blocked": "s",
        "off_target": "x",
        "chance": "D",
    }

    for (x, y), outcome in zip(shot_pitchxy, outcomes):
        c = _colours.get(outcome, "#ffffff")
        m = _markers.get(outcome, "o")
        ax.plot(x, y, marker=m, color=c, markersize=8, markeredgecolor="black", markeredgewidth=0.5)

    # Legend
    handles = [
        mpatches.Patch(color=c, label=k.replace("_", " ").title())
        for k, c in _colours.items()
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.7)

    return fig, ax
