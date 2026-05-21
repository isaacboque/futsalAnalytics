"""
Standalone field calibration script.

Usage
-----
    python scripts/calibrate.py              # prompts for source
    python scripts/calibrate.py video.mp4   # local file
    python scripts/calibrate.py https://... # YouTube URL

The calibrated 6-point array is saved to ``calibration_points.npy`` in the
current working directory.

After installing the package (``pip install -e .``) you can also run:
    futsal-calibrate
"""

import sys
from pathlib import Path

# Allow running as a plain script without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from futsal_analytics.calibration import run_standalone

if __name__ == "__main__":
    run_standalone()
