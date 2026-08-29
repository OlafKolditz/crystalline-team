#!/usr/bin/env python3
"""Joint sweep of pipe fraction, dispersivity, and deep recovery."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import meshio
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_SCRIPT = (
    ROOT / "case6_4pipes_deep_reservoir_feedback" / "run_deep_feedback.py"
)
FLOW_SCRIPT = (
    ROOT
    / "case6_4pipes_temperature_constrained_flow_sweep"
    / "run_flow_fraction_sweep.py"
)
HELPER_SCRIPT = (
    ROOT / "case6_4pipes_deep_release_time_sweep" / "run_tau_sweep.py"
)
OBSERVED = ROOT / "observed_tracer_ZK203_ZK208.csv"
PYTHON = Path("/home/zhai/ogs-env/bin/python")
BASE_PIPE_FRACTION = 2.0 / 7.0
DEEP_RELEASE_DAYS = 60.0
TARGET_GROSS_RECOVERY_KG = 277.6416989672673
MAX_PARALLEL_RUNS = 3
TOLERANCE_PPB = 0.10
MAX_ITERATIONS = 22


def parameter_design() -> list[dict]:
    """Return 16 reproducible LHS points plus four physical anchors."""
    count = 16
    rng = np.random.default_rng(20260729)
    sample = np.empty((count, 3))
    for dim in range(3):
        sample[:, dim] = (
            rng.permutation(count) + rng.random(count)
        ) / count
    f_min, f_max = BASE_PIPE_FRACTION, 0.50
    points = []
    for idx, unit in enumerate(sample, start=1):
        pipe_fraction = f_min + (f_max - f_min) * unit[0]
        alpha = math.exp(math.log(2.0) + math.log(10.0) * unit[1])
        recovery = 0.40 + 0.40 * unit[2]
        points.append(
            {
                "case": f"lhs_{idx:02d}",
                "f_pipe": pipe_fraction,
                "alphaL_m": alpha,
                "Rdeep": recovery,
                "design_type": "LHS",
            }
        )
    points += [
        {
            "case": "anchor_60C",
            "f_pipe": 2.0 / 7.0,
            "alphaL_m": 2.0,
            "Rdeep": 0.80,
            "design_type": "anchor",
        },
        {
            "case": "anchor_80C",
            "f_pipe": 1.0 / 3.0,
            "alphaL_m": 5.0,
            "Rdeep": 0.70,
            "design_type": "anchor",
        },
        {
            "case": "anchor_100C",
            "f_pipe": 0.40,
            "alphaL_m": 10.0,
            "Rdeep": 0.60,
            "design_type": "anchor",
        },
        {
            "case": "anchor_120C",
            "f_pipe": 0.50,
            "alphaL_m": 20.0,
            "Rdeep": 0.40,
            "design_type": "anchor",
        },
    ]
    return points


DESIGN = parameter_design()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def design_case(name: str) -> dict:
    return next(item for item in DESIGN if item["case"] == name)


def inferred_pipe_temperature(pipe_fraction: float) -> float:
    """200 C hot water plus pipe water giving 160 C production."""
    return 200.0 - 40.0 / pipe_fraction


def scale_aperture(path: Path, factor: float) -> None:
    mesh = meshio.read(path)
    if "Aperture" not in mesh.cell_data:
        raise RuntimeError(f"Aperture field not found in {path}")
    mesh.cell_data["Aperture"] = [
        np.asarray(values, dtype=float) * factor
        for values in mesh.cell_data["Aperture"]
    ]
    meshio.write(path, mesh, binary=True)


def run_single(name: str) -> None:
    point = design_case(name)
    pipe_fraction = float(point["f_pipe"])
    deep_fraction = 1.0 - pipe_fraction
    aperture_scale = pipe_fraction / BASE_PIPE_FRACTION
    model = load_module(MODEL_SCRIPT, f"deep_feedback_{name}")
    flow = load_module(FLOW_SCRIPT, f"flow_helpers_{name}")
    helpers = load_module(HELPER_SCRIPT, f"tau_helpers_{name}")
    folder = HERE / "runs" / name
    folder.mkdir(parents=True, exist_ok=True)

    model.HERE = folder
    model.PROJECT = folder / f"case6_joint_{name}.prj"
    model.INPUT = folder / "input_mesh"
    model.OUTPUT = folder / "output"
    model.CURVES = folder / "iteration_curves"
    model.LOGS = folder / "iteration_logs"
    model.PREFIX = f"case6_joint_{name}"
    model.PIPE_FRACTION = pipe_fraction
    model.DEEP_FRACTION = deep_fraction
    model.DEEP_EVENTUAL_RECOVERY = float(point["Rdeep"])
    model.DEEP_RELEASE_DAYS = DEEP_RELEASE_DAYS
    model.MAX_ITERATIONS = MAX_ITERATIONS
    model.TOLERANCE_PPB = TOLERANCE_PPB

    original_prepare = model.prepare_project

    def prepare_joint_case():
        tree = original_prepare()
        helpers.set_fast_dispersivity(tree, float(point["alphaL_m"]))
        flow.set_allocation_curves(model, tree, deep_fraction)
        scale_aperture(
            model.INPUT / "four_pipes_with_storage.vtu",
            aperture_scale,
        )
        return tree

    model.prepare_project = prepare_joint_case
    model.main()


def launch_case(point: dict) -> str:
    name = str(point["case"])
    folder = HERE / "runs" / name
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "launcher.log").open("w", encoding="utf-8") as stream:
        subprocess.run(
            [
                str(PYTHON),
                str(Path(__file__).resolve()),
                "--single",
                name,
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return name


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def observed() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows = {"ZK208": [], "ZK203": []}
    with OBSERVED.open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["well"] in rows:
                rows[row["well"]].append(
                    (
                        float(row["days_since_tracer_injection"]),
                        max(float(row["concentration_ppb"]), 0.0),
                    )
                )
    return {
        well: (
            np.asarray([item[0] for item in values]),
            np.asarray([item[1] for item in values]),
        )
        for well, values in rows.items()
    }


def tracer_timeseries(folder: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    values = {"ZK208": [], "ZK203": []}
    for row in read_csv(folder / "tracer_timeseries_ppb.csv"):
        values[row["well"]].append(
            (float(row["time_days"]), float(row["concentration_ppb"]))
        )
    return {
        well: (
            np.asarray([item[0] for item in series]),
            np.asarray([item[1] for item in series]),
        )
        for well, series in values.items()
    }


def collect_results() -> list[dict]:
    helpers = load_module(HELPER_SCRIPT, "tau_helpers_collect_joint")
    obs = observed()
    observed_width = {
        well: helpers.fwhm(time_days, concentration)
        for well, (time_days, concentration) in obs.items()
    }
    target_peak = {
        "ZK208": (5.0, 518.88),
        "ZK203": (32.0, 164.602),
    }
    rows = []
    for point in DESIGN:
        name = str(point["case"])
        folder = HERE / "runs" / name
        tracer = {
            row["well"]: row
            for row in read_csv(folder / "tracer_comparison_metrics.csv")
        }
        series = tracer_timeseries(folder)
        mass = read_csv(folder / "circulation_mass_summary.csv")[0]
        pressure = {
            row["well"]: float(row["pressure_RMSE_MPa_first_60d"])
            for row in read_csv(folder / "pressure_preservation.csv")
        }
        history = read_csv(folder / "iteration_history.csv")
        feedback = float(history[-1]["maximum_curve_change_ppb"])
        pipe_fraction = float(point["f_pipe"])
        row = {
            **point,
            "f_deep": 1.0 - pipe_fraction,
            "implied_pipe_temperature_C": inferred_pipe_temperature(
                pipe_fraction
            ),
            "aperture_scale": pipe_fraction / BASE_PIPE_FRACTION,
            "tau_deep_days": DEEP_RELEASE_DAYS,
            "iterations": int(history[-1]["iteration"]),
            "final_feedback_change_ppb": feedback,
            "strictly_converged": feedback < TOLERANCE_PPB,
            "gross_produced_mass_238d_kg": float(
                mass["simulated_gross_produced_mass_238d_kg"]
            ),
            "deep_inventory_238d_kg": float(
                mass["recoverable_deep_inventory_at_238d_kg"]
            ),
        }
        row["gross_mass_error_kg"] = (
            row["gross_produced_mass_238d_kg"] - TARGET_GROSS_RECOVERY_KG
        )
        for well in ("ZK403", "ZK208", "ZK203"):
            row[f"{well}_pressure_RMSE_MPa"] = pressure[well]
        for well in ("ZK208", "ZK203"):
            time_days, concentration = series[well]
            peak_day = float(tracer[well]["peak_day"])
            row[f"{well}_peak_day"] = peak_day
            row[f"{well}_peak_ppb"] = float(tracer[well]["peak_ppb"])
            row[f"{well}_FWHM_days"] = helpers.fwhm(
                time_days, concentration
            )
            row[f"{well}_observed_FWHM_days"] = observed_width[well]
            row[f"{well}_tail_integral_ppb_days"] = helpers.tail_integral(
                time_days, concentration, peak_day
            )
            row[f"{well}_RMSE_ppb"] = float(tracer[well]["RMSE_ppb_238d"])

        row["curve_score"] = 0.5 * sum(
            row[f"{well}_RMSE_ppb"] / target_peak[well][1]
            for well in ("ZK208", "ZK203")
        )
        row["peak_time_score"] = 0.5 * sum(
            abs(row[f"{well}_peak_day"] - target_peak[well][0])
            / target_peak[well][0]
            for well in ("ZK208", "ZK203")
        )
        row["peak_height_score"] = 0.5 * sum(
            abs(row[f"{well}_peak_ppb"] - target_peak[well][1])
            / target_peak[well][1]
            for well in ("ZK208", "ZK203")
        )
        row["width_score"] = 0.5 * sum(
            abs(row[f"{well}_FWHM_days"] - observed_width[well])
            / observed_width[well]
            for well in ("ZK208", "ZK203")
        )
        row["mass_score"] = (
            abs(row["gross_mass_error_kg"]) / TARGET_GROSS_RECOVERY_KG
        )
        row["pressure_score"] = max(
            row[f"{well}_pressure_RMSE_MPa"]
            for well in ("ZK403", "ZK208", "ZK203")
        )
        row["balanced_score"] = sum(
            row[key]
            for key in (
                "curve_score",
                "peak_time_score",
                "peak_height_score",
                "width_score",
                "mass_score",
                "pressure_score",
            )
        )
        rows.append(row)
    return sorted(rows, key=lambda item: item["balanced_score"])


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_results(rows: list[dict]) -> None:
    write_csv(HERE / "joint_sweep_summary_ranked.csv", rows)
    write_csv(HERE / "joint_sweep_top5.csv", rows[:5])

    obs = observed()
    fig, axes = plt.subplots(
        2, 1, figsize=(11.0, 7.7), sharex=True, constrained_layout=True
    )
    colors = plt.cm.viridis(np.linspace(0.05, 0.90, 5))
    for rank, (row, color) in enumerate(zip(rows[:5], colors), start=1):
        series = tracer_timeseries(HERE / "runs" / str(row["case"]))
        for ax, well in zip(axes, ("ZK208", "ZK203")):
            time_days, concentration = series[well]
            ax.plot(
                time_days,
                concentration,
                color=color,
                lw=1.55,
                label=(
                    f"#{rank} {row['case']}: "
                    rf"$f_p$={100 * row['f_pipe']:.1f}%, "
                    rf"$\alpha_L$={row['alphaL_m']:.1f} m, "
                    rf"$R_d$={100 * row['Rdeep']:.0f}%"
                ),
            )
    for ax, well in zip(axes, ("ZK208", "ZK203")):
        obs_t, obs_c = obs[well]
        ax.scatter(
            obs_t,
            obs_c,
            s=28,
            facecolors="none",
            edgecolors="black",
            lw=0.85,
            label="Observed",
        )
        ax.set_title(well, loc="left")
        ax.set_ylabel("Tracer concentration (ppb)")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set(xlabel="Time since tracer injection (days)", xlim=(0, 238))
    fig.suptitle("Top five joint-calibration cases")
    fig.savefig(HERE / "joint_sweep_top5_tracer.png", dpi=220)
    fig.savefig(HERE / "joint_sweep_top5_tracer.svg")
    plt.close(fig)

    fig, axes = plt.subplots(
        1, 3, figsize=(13.8, 4.5), constrained_layout=True
    )
    values = (
        ("f_pipe", r"$f_{pipe+storage}$ (%)", 100.0),
        ("alphaL_m", r"$\alpha_L$ (m)", 1.0),
        ("Rdeep", r"$R_{deep}$ (%)", 100.0),
    )
    scores = np.asarray([row["balanced_score"] for row in rows])
    for ax, (key, label, scale) in zip(axes, values):
        scatter = ax.scatter(
            [scale * row[key] for row in rows],
            scores,
            c=scores,
            cmap="viridis_r",
            s=42,
        )
        ax.set(xlabel=label, ylabel="Balanced score")
        ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes, label="Balanced score")
    fig.savefig(HERE / "joint_sweep_parameter_response.png", dpi=220)
    fig.savefig(HERE / "joint_sweep_parameter_response.svg")
    plt.close(fig)

    best = rows[0]
    components = (
        "curve_score",
        "peak_time_score",
        "peak_height_score",
        "width_score",
        "mass_score",
        "pressure_score",
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
    x = np.arange(5)
    bottom = np.zeros(5)
    for component in components:
        values = np.asarray([row[component] for row in rows[:5]])
        ax.bar(x, values, bottom=bottom, label=component.replace("_", " "))
        bottom += values
    ax.set(
        xticks=x,
        xticklabels=[str(row["case"]) for row in rows[:5]],
        ylabel="Score contribution",
        title="Top-five objective decomposition",
    )
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(HERE / "joint_sweep_score_components.png", dpi=220)
    fig.savefig(HERE / "joint_sweep_score_components.svg")
    plt.close(fig)

    lines = [
        "# Three-parameter joint sweep",
        "",
        f"Cases: {len(rows)} (16 Latin-hypercube points + 4 anchors).",
        f"Deep release time: {DEEP_RELEASE_DAYS:g} days.",
        "Permeability is fixed. Aperture is scaled deterministically as "
        "A/A0=fpipe/(2/7).",
        f"Feedback tolerance: {TOLERANCE_PPB:g} ppb.",
        "",
        f"Best coarse case: {best['case']}.",
        f"- fpipe+storage: {best['f_pipe']:.4%}.",
        f"- implied pipe temperature: "
        f"{best['implied_pipe_temperature_C']:.2f} C.",
        f"- alphaL: {best['alphaL_m']:.4g} m.",
        f"- Rdeep: {best['Rdeep']:.4%}.",
        f"- gross recovery: {best['gross_produced_mass_238d_kg']:.3f} kg "
        f"(target {TARGET_GROSS_RECOVERY_KG:.3f} kg).",
        f"- ZK208 peak: {best['ZK208_peak_day']:.3f} d, "
        f"{best['ZK208_peak_ppb']:.3f} ppb; "
        f"FWHM={best['ZK208_FWHM_days']:.3f} d; "
        f"RMSE={best['ZK208_RMSE_ppb']:.3f} ppb.",
        f"- ZK203 peak: {best['ZK203_peak_day']:.3f} d, "
        f"{best['ZK203_peak_ppb']:.3f} ppb; "
        f"FWHM={best['ZK203_FWHM_days']:.3f} d; "
        f"RMSE={best['ZK203_RMSE_ppb']:.3f} ppb.",
        f"- maximum pressure RMSE: {best['pressure_score']:.5f} MPa.",
        f"- balanced score: {best['balanced_score']:.5f}.",
        "",
        "The balanced score is the unweighted sum of normalized full-curve "
        "RMSE, peak-time error, peak-height error, FWHM error, cumulative-"
        "mass error, and maximum pressure RMSE in MPa.",
    ]
    (HERE / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_design() -> None:
    rows = []
    for point in DESIGN:
        pipe_fraction = float(point["f_pipe"])
        rows.append(
            {
                **point,
                "f_deep": 1.0 - pipe_fraction,
                "implied_pipe_temperature_C": inferred_pipe_temperature(
                    pipe_fraction
                ),
                "aperture_scale": pipe_fraction / BASE_PIPE_FRACTION,
            }
        )
    write_csv(HERE / "joint_parameter_design.csv", rows)


def orchestrate() -> None:
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(OBSERVED, HERE / OBSERVED.name)
    write_design()
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_RUNS) as executor:
        futures = {
            executor.submit(launch_case, point): str(point["case"])
            for point in DESIGN
        }
        for future in as_completed(futures):
            print(f"Completed {future.result()}", flush=True)
    rows = collect_results()
    write_results(rows)
    print((HERE / "README.md").read_text(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single")
    args = parser.parse_args()
    if args.single:
        run_single(args.single)
    else:
        orchestrate()


if __name__ == "__main__":
    main()
