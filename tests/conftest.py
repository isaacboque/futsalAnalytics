"""Shared pytest fixtures."""

import numpy as np
import pytest


@pytest.fixture()
def synthetic_frame():
    """640×480 BGR frame filled with a solid green colour (simulates a pitch)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (34, 139, 34)
    return frame


@pytest.fixture()
def default_polygon():
    """Unit-square polygon scaled to 640×480 frame with 10 % margin."""
    w, h = 640, 480
    mx, my = int(w * 0.1), int(h * 0.15)
    cx = w // 2
    return np.array(
        [
            [mx, my],
            [cx, my],
            [w - mx, my],
            [w - mx, h - my],
            [cx, h - my],
            [mx, h - my],
        ],
        dtype=np.float32,
    )
