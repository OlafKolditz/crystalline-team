#!/usr/bin/env python3
"""Joint permeability sweep for active-well pressure and monitoring head change."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CALIBRATION_DIR = HERE.parent
CASE_DIR = CALIBRATION_DIR.parent
SHARED_DIR = CASE_DIR.parent / "case1_k9_2e-11_k10_4e-13"
BASE_PROJECT = CALIBRATION_DIR / "refined_sweep" / "best_calibrated.prj"
WORKBOOK = CASE_DIR / "monitoring" / "yangyi_after_20190113_for_ogs.xlsx"
MONITORING = (
    CASE_DIR
    / "monitoring"
    / "recommended_monitoring_points_all7_ZK501_bedrock_noMatrix.csv"
)
PROJECTS = HERE / "projects"
RESULTS = HERE / "results"
TEMP_ROOT = Path("/tmp/yangyi_joint_downhole_head_calibration")
DEFAULT_OGS = Path("/home/zhai/ogs-env/bin/ogs")
DEFAULT_SEED = 20260720

PRESSURE_WELLS = ("ZK403", "ZK208", "ZK203")
HEAD_WELLS = ("SC211", "ZK207", "ZK206")
PRESSURE_SCALES_MPA = {"ZK403": 0.250, "ZK208": 0.269, "ZK203": 0.200}
HEAD_SCALES_M = {"SC211": 8.235, "ZK207": 6.892, "ZK206": 4.143}
PRESSURE_GUARDRAIL_MPA = 0.5
OBJECTIVE_VERSION = "joint_v1_observed_scale_equal_groups"

# Bounds are log-uniform. Materials 1, 5, and 12 are the active-well
# completions; Materials 4, 9, and 10 intersect the requested monitoring wells.
PARAMETERS = {
    1: (7.0e-9, 2.0e-8, "ZK208 completion"),
    5: (7.5e-11, 1.4e-10, "ZK403 completion"),
    12: (2.0e-12, 1.2e-11, "ZK203/SC211/ZK207 low-res"),
    9: (1.0e-11, 1.0e-10, "ZK207 fault"),
}
BASELINE = {
    1: 8.55527025293163e-9,
    5: 9.751692494468547e-11,
    12: 1.0935842852282497e-11,
    9: 2.0e-11,
}
INFORMED = {
    1: BASELINE[1],
    5: BASELINE[5],
    12: BASELINE[12],
    9: 8.741659411746023e-11,
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ACTIVE_PLOT = load_module("joint_active_plot", SHARED_DIR / "plot_wellhead_pressure.py")
ACTIVE_COMPARE = load_module(
    "joint_active_compare", SHARED_DIR / "compare_observed_wellhead_pressure.py"
)
HEAD_COMPARE = load_module(
    "joint_head_compare", CASE_DIR / "compare_simulated_observed_head_change.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=36, help="number of Latin-hypercube random trials")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ogs", type=Path, default=DEFAULT_OGS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_one_row(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    converted: dict[str, object] = {}
    for key, value in row.items():
        if value in ("True", "False"):
            converted[key] = value == "True"
            continue
        try:
            converted[key] = float(value)
        except (TypeError, ValueError):
            converted[key] = value
    return converted


def latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = np.empty((count, dimensions), dtype=float)
    for column in range(dimensions):
        samples[:, column] = (rng.permutation(count) + rng.random(count)) / count
    return samples


def trials(count: int, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"case_id": "baseline_000", "is_baseline": True, "is_informed": False},
        {"case_id": "informed_001", "is_baseline": False, "is_informed": True},
    ]
    for row, values in zip(rows, (BASELINE, INFORMED)):
        row.update({f"k{mid}_m2": value for mid, value in values.items()})
        row["random_seed"] = seed

    mids = list(PARAMETERS)
    samples = latin_hypercube(count, len(mids), seed)
    for index, sample in enumerate(samples, start=1):
        row: dict[str, object] = {
            "case_id": f"joint_{index:03d}",
            "random_seed": seed,
            "is_baseline": False,
            "is_informed": False,
        }
        for fraction, mid in zip(sample, mids):
            low, high, _ = PARAMETERS[mid]
            row[f"k{mid}_m2"] = 10.0 ** (
                math.log10(low) + fraction * (math.log10(high) - math.log10(low))
            )
        rows.append(row)
    return rows


def permeability_value(medium: ET.Element) -> ET.Element:
    for prop in medium.findall("./properties/property"):
        if prop.findtext("name") == "permeability":
            value = prop.find("value")
            if value is not None:
                return value
    raise RuntimeError(f"Medium {medium.attrib.get('id')} has no permeability")


def create_projects(rows: list[dict[str, object]]) -> None:
    PROJECTS.mkdir(parents=True, exist_ok=True)
    for row in rows:
        tree = ET.parse(BASE_PROJECT)
        root = tree.getroot()
        media = {int(m.attrib["id"]): m for m in root.findall(".//medium")}
        for mid in PARAMETERS:
            permeability_value(media[mid]).text = f"{float(row[f'k{mid}_m2']):.12e}"
        prefix = root.find(".//time_loop/output/prefix")
        if prefix is None:
            raise RuntimeError("Output prefix missing")
        prefix.text = str(row["case_id"])
        variables = root.find(".//time_loop/output/variables")
        if variables is not None:
            for variable in list(variables):
                if variable.text != "pressure":
                    variables.remove(variable)
        ET.indent(tree, space="  ")
        tree.write(PROJECTS / f"{row['case_id']}.prj", encoding="UTF-8", xml_declaration=True)


def head_metrics(comparison: pd.DataFrame) -> list[dict[str, object]]:
    selected = comparison[comparison["well"].isin(HEAD_WELLS)].copy()
    averaged = (
        selected.groupby(["well", "Date", "time_days_from_20190113"], as_index=False)
        .agg(observed_delta_h_m=("obs_delta_h_m", "first"), simulated_delta_h_m=("sim_delta_h_m", "mean"))
    )
    averaged["residual_m"] = averaged["simulated_delta_h_m"] - averaged["observed_delta_h_m"]
    rows = []
    for well, group in averaged.groupby("well"):
        residual = group["residual_m"].to_numpy(float)
        rows.append(
            {
                "well": well,
                "n": len(group),
                "bias_m": float(np.mean(residual)),
                "mae_m": float(np.mean(np.abs(residual))),
                "rmse_m": float(np.sqrt(np.mean(residual**2))),
                "max_abs_error_m": float(np.max(np.abs(residual))),
                "final_observed_m": float(group.iloc[-1]["observed_delta_h_m"]),
                "final_simulated_m": float(group.iloc[-1]["simulated_delta_h_m"]),
            }
        )
    return rows


def plot_target_heads(comparison: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = comparison[comparison["well"].isin(HEAD_WELLS)].copy()
    averaged = (
        selected.groupby(["well", "time_days_from_20190113"], as_index=False)
        .agg(observed_delta_h_m=("obs_delta_h_m", "first"), simulated_delta_h_m=("sim_delta_h_m", "mean"))
    )
    colors = {"SC211": "#1f77b4", "ZK207": "#d62728", "ZK206": "#2ca02c"}
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    for axis, well in zip(axes, HEAD_WELLS):
        group = averaged[averaged["well"] == well].sort_values("time_days_from_20190113")
        axis.plot(
            group["time_days_from_20190113"],
            group["observed_delta_h_m"],
            marker="o",
            color=colors[well],
            linewidth=2,
            label="Observed",
        )
        axis.plot(
            group["time_days_from_20190113"],
            group["simulated_delta_h_m"],
            linestyle="--",
            color="black",
            linewidth=1.8,
            label="Simulated (mapped-point mean)",
        )
        axis.axhline(0, color="0.5", linewidth=0.8)
        axis.set_title(well)
        axis.set_ylabel("Head change (m)")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("Time from 2019-01-13 (days)")
    fig.suptitle("Joint-calibration targets: observed vs simulated hydraulic-head variation")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def score(active: list[dict[str, object]], heads: list[dict[str, object]]) -> dict[str, float]:
    active_by_well = {str(row["well"]): row for row in active}
    head_by_well = {str(row["well"]): row for row in heads}
    pressure_rmse = {
        well: float(active_by_well[well]["absolute_RMSE_MPa"])
        for well in PRESSURE_WELLS
    }
    head_rmse = {well: float(head_by_well[well]["rmse_m"]) for well in HEAD_WELLS}
    pressure_normalized = [
        pressure_rmse[well] / PRESSURE_SCALES_MPA[well] for well in PRESSURE_WELLS
    ]
    head_normalized = [head_rmse[well] / HEAD_SCALES_M[well] for well in HEAD_WELLS]
    pressure_score = float(np.sqrt(np.mean(np.square(pressure_normalized))))
    head_score = float(np.sqrt(np.mean(np.square(head_normalized))))
    guardrail_pass = all(value <= PRESSURE_GUARDRAIL_MPA for value in pressure_rmse.values())
    return {
        "objective_version": OBJECTIVE_VERSION,
        "joint_objective": float(np.sqrt(0.5 * pressure_score**2 + 0.5 * head_score**2)),
        "pressure_group_score": pressure_score,
        "head_group_score": head_score,
        "worst_normalized_RMSE": float(max(pressure_normalized + head_normalized)),
        "pressure_guardrail_pass": guardrail_pass,
        **{f"pressure_RMSE_{well}_MPa": value for well, value in pressure_rmse.items()},
        **{f"head_RMSE_{well}_m": value for well, value in head_rmse.items()},
    }


def evaluate(
    trial: dict[str, object],
    ogs: Path,
    active_observations,
    active_dates,
    head_observations: pd.DataFrame,
    resume: bool,
    force: bool,
) -> dict[str, object]:
    case_id = str(trial["case_id"])
    result_dir = RESULTS / case_id
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_dir / "summary.csv"
    if resume and not force and summary_path.exists():
        saved = read_one_row(summary_path)
        if saved.get("status") == "ok" and saved.get("objective_version") == OBJECTIVE_VERSION:
            return {**trial, **saved}

    temp_dir = TEMP_ROOT / case_id
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    started = time.monotonic()
    result: dict[str, object] = {}
    try:
        with (result_dir / "ogs.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [str(ogs), str(PROJECTS / f"{case_id}.prj"), "-m", str(CASE_DIR), "-o", str(temp_dir)],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"OGS returned {completed.returncode}")

        wells = ACTIVE_PLOT.read_wells(CASE_DIR)
        datasets = ACTIVE_PLOT.read_pvd(temp_dir / f"{case_id}.pvd")
        active_sim = ACTIVE_PLOT.extract(datasets, wells, 1000.0, 9.81)
        active_csv = result_dir / "active_pressure_timeseries.csv"
        ACTIVE_PLOT.write_csv(active_csv, active_sim)
        active_by_day = ACTIVE_COMPARE.read_simulation(active_csv)
        active_comp = ACTIVE_COMPARE.build_comparison(active_by_day, active_observations, active_dates)
        active_summary = ACTIVE_COMPARE.metrics(active_comp)
        write_csv(result_dir / "active_pressure_metrics.csv", active_summary)

        head_results = HEAD_COMPARE.parse_pvd(temp_dir / f"{case_id}.pvd")
        head_sim = HEAD_COMPARE.extract_sim(head_results, MONITORING, 1000.0, 9.81, 86400.0, None)
        zero_day = float(head_observations["time_days"].min())
        head_zero = HEAD_COMPARE.add_zeroed_delta(head_sim, zero_day)
        head_comp = HEAD_COMPARE.interpolate_to_obs(head_zero, head_observations)
        head_summary = head_metrics(head_comp)
        write_csv(result_dir / "monitoring_head_metrics.csv", head_summary)

        result.update(score(active_summary, head_summary))
        result.update({"status": "ok", "ogs_returncode": 0})
    except Exception as exc:
        result.update({"status": "failed", "error": str(exc)})
    finally:
        result["runtime_s"] = time.monotonic() - started
        shutil.rmtree(temp_dir, ignore_errors=True)

    write_csv(summary_path, [{**trial, **result}])
    return {**trial, **result}


def make_best_outputs(best: dict[str, object], ogs: Path, head_observations: pd.DataFrame) -> None:
    case_id = str(best["case_id"])
    best_project = HERE / "best_joint_calibrated.prj"
    shutil.copy2(PROJECTS / f"{case_id}.prj", best_project)
    output_dir = HERE / "best_case_output"
    comparison_dir = HERE / "best_case_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    with (output_dir / "ogs.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [str(ogs), str(best_project), "-m", str(CASE_DIR), "-o", str(output_dir)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Best-case OGS run failed; see {output_dir / 'ogs.log'}")

    pvd = output_dir / f"{case_id}.pvd"
    wells = ACTIVE_PLOT.read_wells(CASE_DIR)
    active_sim = ACTIVE_PLOT.extract(ACTIVE_PLOT.read_pvd(pvd), wells, 1000.0, 9.81)
    active_csv = comparison_dir / "active_pressure_timeseries.csv"
    ACTIVE_PLOT.write_csv(active_csv, active_sim)
    active_obs, active_dates = ACTIVE_COMPARE.read_observations(WORKBOOK)
    active_comp = ACTIVE_COMPARE.build_comparison(
        ACTIVE_COMPARE.read_simulation(active_csv), active_obs, active_dates
    )
    active_summary = ACTIVE_COMPARE.metrics(active_comp)
    write_csv(comparison_dir / "active_pressure_comparison.csv", active_comp)
    write_csv(comparison_dir / "active_pressure_metrics.csv", active_summary)
    ACTIVE_COMPARE.make_completion_plot(comparison_dir / "downhole_pressure_comparison.png", active_comp)
    ACTIVE_COMPARE.make_combined_completion_plot(
        comparison_dir / "combined_downhole_pressure_comparison.png", active_comp
    )

    head_results = HEAD_COMPARE.parse_pvd(pvd)
    head_sim = HEAD_COMPARE.extract_sim(head_results, MONITORING, 1000.0, 9.81, 86400.0, None)
    zero_day = float(head_observations["time_days"].min())
    head_zero = HEAD_COMPARE.add_zeroed_delta(head_sim, zero_day)
    head_comp = HEAD_COMPARE.interpolate_to_obs(head_zero, head_observations)
    head_summary = head_metrics(head_comp)
    head_zero.to_csv(comparison_dir / "monitoring_head_change_all_outputs.csv", index=False)
    head_comp.to_csv(comparison_dir / "monitoring_head_comparison.csv", index=False)
    write_csv(comparison_dir / "monitoring_head_metrics.csv", head_summary)
    HEAD_COMPARE.plot_well_comparison(head_comp, comparison_dir, zero_day)
    HEAD_COMPARE.plot_all_mean(head_comp, comparison_dir / "monitoring_head_comparison.png", zero_day)
    plot_target_heads(head_comp, comparison_dir / "target_monitoring_head_comparison.png")
    write_csv(HERE / "best_joint_parameters_and_metrics.csv", [best])


def main() -> None:
    args = parse_args()
    if args.cases < 1 or args.workers < 1:
        raise SystemExit("--cases and --workers must be positive")
    if not args.ogs.is_file():
        raise SystemExit(f"OGS executable not found: {args.ogs}")
    PROJECTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    all_trials = trials(args.cases, args.seed)
    write_csv(HERE / "joint_parameter_settings.csv", all_trials)
    create_projects(all_trials)
    active_observations, active_dates = ACTIVE_COMPARE.read_observations(WORKBOOK)
    head_observations = HEAD_COMPARE.read_observed_from_xlsx(
        WORKBOOK, "Monitor_compare_after20190113"
    )

    rows: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                evaluate,
                trial,
                args.ogs.resolve(),
                active_observations,
                active_dates,
                head_observations,
                args.resume,
                args.force,
            ): str(trial["case_id"])
            for trial in all_trials
        }
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['case_id']}: {row.get('status')} "
                f"objective={row.get('joint_objective', float('nan')):.4f} "
                f"runtime={row.get('runtime_s', float('nan')):.1f}s",
                flush=True,
            )

    successful = [row for row in rows if row.get("status") == "ok"]
    successful.sort(
        key=lambda row: (
            not bool(row.get("pressure_guardrail_pass")),
            float(row["joint_objective"]),
        )
    )
    for rank, row in enumerate(successful, start=1):
        row["rank"] = rank
    failed = [row for row in rows if row.get("status") != "ok"]
    write_csv(HERE / "ranking_all_cases.csv", successful + failed)
    write_csv(HERE / "top_10_cases.csv", successful[:10])
    if not successful:
        raise RuntimeError("No successful calibration cases")

    make_best_outputs(successful[0], args.ogs.resolve(), head_observations)
    print(f"Best case: {successful[0]['case_id']}")
    print(f"Joint objective: {successful[0]['joint_objective']:.6f}")
    print(f"Outputs: {HERE}")


if __name__ == "__main__":
    main()
