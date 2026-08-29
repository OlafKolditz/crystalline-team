#!/usr/bin/env python3
"""Run the second, narrower active-well permeability calibration sweep."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))

import run_random_calibration as calibration  # noqa: E402


# Round-one evidence places the useful ZK403 range near the original lower
# bound, while ZK208 and ZK203 favor moderately higher k1 and k12 values.
calibration.ROOT = HERE
calibration.PROJECTS = HERE / "projects"
calibration.RESULTS = HERE / "results"
calibration.TEMP_ROOT = Path("/tmp/yangyi_wellhead_refined_calibration")
calibration.DEFAULT_SEED = 20260720
calibration.PARAMETERS = {
    5: (5.0e-11, 2.0e-10, "ZK403"),
    1: (7.0e-9, 3.0e-8, "ZK208"),
    12: (4.0e-12, 2.0e-11, "ZK203"),
}

# The first-round winner is the baseline for this refinement.
calibration.BASELINE = {
    5: 1.593264259268089e-10,
    1: 3.2053336218730725e-9,
    12: 3.0003109292623097e-12,
}


if __name__ == "__main__":
    calibration.main()
