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
# date = "2025-04-28"
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

import matplotlib.pyplot as plt
import numpy as np

from meshing import run_meshing
from ogs_runner import run_ogs_simulation
from postprocessing import run_postprocessing
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


def flow_rate_total(time_values):
    time_values = np.asarray(time_values, dtype=float)
    conditions = [
        time_values < FLOW_TIMES[0],
        time_values < FLOW_TIMES[1],
        time_values < FLOW_TIMES[2],
        time_values < FLOW_TIMES[3],
        time_values < FLOW_TIMES[4],
        time_values < FLOW_TIMES[5],
        time_values < FLOW_TIMES[6],
        time_values < FLOW_TIMES[7],
    ]
    return np.select(conditions, FLOW_VALUES, default=0.0)


def input_flux(time_values):
    return flow_rate_total(time_values) / BOREHOLE_LENGTH


def plot_input_flux_data() -> None:
    time_values = np.linspace(0.0, 3000.0, 1000)
    total_flow = flow_rate_total(time_values)
    flux = input_flux(time_values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    axes[0].step(time_values, total_flow, where="post", linewidth=2.0)
    axes[0].set_title("Total Flow Rate")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Flow rate [m^3/s]")

    axes[1].step(time_values, flux, where="post", linewidth=2.0)
    axes[1].set_title("Input Flux For OGS")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Flux [m^2/s]")

    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.set_xlim(0.0, 3000.0)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.tight_layout()
    plt.show()
    plt.close(fig)


# %% [markdown]
# ### Preprocessing: Input Flux Plot

# %%
plot_input_flux_data()


# %% [markdown]
# ### Step 1: Meshing

# %%
input_dir = Path(INPUT_DIR).resolve()
output_dir = Path(OUTPUT_DIR).resolve() if OUTPUT_DIR is not None else input_dir / "_out"
mesh_file = find_mesh_file(input_dir)
borehole_seed_file = find_borehole_seed_file(input_dir)
project_file = find_project_file(input_dir)
mesh_output_name = None

if project_file is not None:
    project_mesh_names = read_project_mesh_names(project_file)
    if project_mesh_names:
        mesh_output_name = project_mesh_names[0]

if mesh_file is None or borehole_seed_file is None:
    meshing_out_dir = None
    print("Skipping meshing because no mesh input files were found.")
else:
    meshing_out_dir = run_meshing(
        orig_dir=input_dir,
        out_dir=output_dir,
        mesh_file=mesh_file,
        borehole_seed_file=borehole_seed_file,
        mesh_output_name=mesh_output_name,
    )

if meshing_out_dir is not None:
    print(f"Meshing output directory: {meshing_out_dir}")


# %% [markdown]
# ### Step 2: Run OGS

# %%
input_dir = Path(INPUT_DIR).resolve()
output_dir = Path(OUTPUT_DIR).resolve() if OUTPUT_DIR is not None else input_dir / "_out"
project_file = find_project_file(input_dir)
existing_pvd_file = find_any_pvd_file(input_dir)
previous_meshing_out_dir = globals().get("meshing_out_dir")
run_out_dir = Path(previous_meshing_out_dir).resolve() if previous_meshing_out_dir is not None else output_dir

if project_file is None:
    pvd_file = find_any_pvd_file(run_out_dir) or existing_pvd_file
    print("Skipping OGS because no project file was found.")
else:
    pvd_file = run_ogs_simulation(
        orig_dir=input_dir,
        out_dir=run_out_dir,
        project_file=project_file,
    )

if pvd_file is not None:
    print(f"OGS result file: {pvd_file}")


# %% [markdown]
# ### Step 3: Postprocessing

# %%
input_dir = Path(INPUT_DIR).resolve()
output_dir = Path(OUTPUT_DIR).resolve() if OUTPUT_DIR is not None else input_dir / "_out"
pvd_file = globals().get("pvd_file")
previous_meshing_out_dir = globals().get("meshing_out_dir")
run_out_dir = Path(previous_meshing_out_dir).resolve() if previous_meshing_out_dir is not None else input_dir
mesh_file = find_mesh_file(input_dir)

if pvd_file is None:
    pvd_file = find_any_pvd_file(run_out_dir)
if pvd_file is None:
    pvd_file = find_any_pvd_file(input_dir)
if pvd_file is None:
    raise FileNotFoundError(f"No PVD result file found under {input_dir}.")

initial_mesh_file = find_mesh_file(run_out_dir) or find_mesh_file(pvd_file.parent) or mesh_file
postprocessing_out_dir = output_dir

run_postprocessing(
    out_dir=postprocessing_out_dir,
    pvd_file=pvd_file,
    initial_mesh_file=initial_mesh_file,
)

print(f"Postprocessing output directory: {postprocessing_out_dir}")
