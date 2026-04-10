# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python (.venv)
#     language: python
#     name: venv
# ---

# %% [raw]
# +++
# title = "OpenGeoSys Workflow for STIMTEC DFN model"
# date = "2026-04-09"
# author = "Mostafa Mollaali"
# # web_subsection = "liquid-flow"
# +++

# %% [markdown]
# ## STIMTEC DFN Workflow
#
# This notebook-style driver simply calls the three separated workflow steps:
#
# - `meshing.py`
# - `ogs_runner.py`
# - `postprocessing.py`

# %%
from pathlib import Path

import numpy as np

from meshing import run_meshing
from ogs_runner import run_ogs_simulation
from postprocessing import run_postprocessing
from preprocessing import FlowRateSchedule, run_preprocessing
from workflow_paths import (
    find_any_pvd_file,
    find_borehole_seed_file,
    find_mesh_file,
    find_project_file,
    read_project_mesh_names,
)


# %%
def _resolve_workflow_dir() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent

    cwd = Path.cwd().resolve()
    if (cwd / "meshing.py").exists() and (cwd / "ogs_runner.py").exists():
        return cwd
    if (cwd / "workflow" / "meshing.py").exists():
        return cwd / "workflow"
    return cwd


SCRIPT_DIR = _resolve_workflow_dir()
INPUT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = None

BOREHOLE_LENGTH = 55.6
FLOW_TIMES = np.array([1794.21, 1909.81, 2079.81, 2160.01, 2330.81, 2430.41, 2517.81, 2669.81], dtype=float)
FLOW_VALUES = np.array([4.99798e-08, 9.31177e-06, 1.90107e-05, 3.41504e-05, 6.76590e-05, 1.01353e-04, 1.33182e-04, 2.52185e-07], dtype=float)
PREPROCESSING_TIME_MAX = 3000.0

FLOW_RATE_SCHEDULE = FlowRateSchedule(
    borehole_length=BOREHOLE_LENGTH,
    flow_times=FLOW_TIMES,
    flow_values=FLOW_VALUES,
    time_max=PREPROCESSING_TIME_MAX,
)


# %% [markdown]
# ### Preprocessing: Input Flux Plot

# %%
run_preprocessing(FLOW_RATE_SCHEDULE)


