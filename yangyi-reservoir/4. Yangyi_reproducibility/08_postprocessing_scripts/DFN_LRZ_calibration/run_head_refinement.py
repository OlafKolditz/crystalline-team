#!/usr/bin/env python3
"""Refine k12/k9 for SC211 and ZK207 while preserving active-well pressure."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent


def load_parent():
    path = PARENT / "run_joint_calibration.py"
    spec = importlib.util.spec_from_file_location("joint_refinement_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


joint = load_parent()

# Keep Materials 1 and 5 fixed at the joint winner. Refine only the two
# permeabilities that control SC211 and the two mapped ZK207 responses.
joint.HERE = HERE
joint.PROJECTS = HERE / "projects"
joint.RESULTS = HERE / "results"
joint.TEMP_ROOT = Path("/tmp/yangyi_joint_head_refinement")
joint.BASE_PROJECT = PARENT / "best_joint_calibrated.prj"
joint.PARAMETERS = {
    12: (1.8e-12, 3.2e-12, "ZK203/SC211/ZK207 low-res"),
    9: (1.0e-11, 3.5e-11, "ZK207 fault"),
}
joint.BASELINE = {12: 2.887007451807084e-12, 9: 3.883308202027928e-11}
joint.INFORMED = {12: 2.263469259981804e-12, 9: 1.6106247632662385e-11}
joint.OBJECTIVE_VERSION = "head_refinement_v1_pressure_constrained"

PRESSURE_LIMITS_MPA = {"ZK403": 0.32, "ZK208": 0.46, "ZK203": 0.45}
TARGET_HEAD_WELLS = ("SC211", "ZK207")


def constrained_head_score(active, heads):
    active_by_well = {str(row["well"]): row for row in active}
    head_by_well = {str(row["well"]): row for row in heads}
    pressure_rmse = {
        well: float(active_by_well[well]["absolute_RMSE_MPa"])
        for well in joint.PRESSURE_WELLS
    }
    head_rmse = {
        well: float(head_by_well[well]["rmse_m"])
        for well in joint.HEAD_WELLS
    }
    pressure_normalized = [
        pressure_rmse[well] / joint.PRESSURE_SCALES_MPA[well]
        for well in joint.PRESSURE_WELLS
    ]
    target_normalized = [
        head_rmse[well] / joint.HEAD_SCALES_M[well] for well in TARGET_HEAD_WELLS
    ]
    pressure_score = float(np.sqrt(np.mean(np.square(pressure_normalized))))
    head_score = float(np.sqrt(np.mean(np.square(target_normalized))))
    guardrail_pass = all(
        pressure_rmse[well] <= PRESSURE_LIMITS_MPA[well]
        for well in joint.PRESSURE_WELLS
    )
    return {
        "objective_version": joint.OBJECTIVE_VERSION,
        "joint_objective": head_score,
        "pressure_group_score": pressure_score,
        "head_group_score": head_score,
        "worst_normalized_RMSE": float(max(pressure_normalized + target_normalized)),
        "pressure_guardrail_pass": guardrail_pass,
        **{f"pressure_RMSE_{well}_MPa": value for well, value in pressure_rmse.items()},
        **{f"head_RMSE_{well}_m": value for well, value in head_rmse.items()},
    }


joint.score = constrained_head_score


if __name__ == "__main__":
    joint.main()

