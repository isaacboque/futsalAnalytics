"""Shared fixtures for PivotIQ tests."""

import numpy as np
import pytest


@pytest.fixture()
def identity_H() -> np.ndarray:
    """3×3 identity matrix — usable as a no-op homography."""
    return np.eye(3, dtype=np.float64)


@pytest.fixture()
def simple_bbox_data() -> dict:
    return {"x1": 10.0, "y1": 20.0, "x2": 50.0, "y2": 80.0}
