#!/usr/bin/env python3
"""Run and rank targeted random permeability trials for the three active wells."""

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


ROOT = Path(__file__).resolve().parent
CASE_DIR = ROOT.parent
SHARED_DIR = CASE_DIR.parent / "case1_k9_2e-11_k10_4e-13"
BASE_PROJECT = CASE_DIR / "c1_k9_2e-11_k10_4e-13_200d.prj"
WORKBOOK = CASE_DIR / "monitoring" / "yangyi_after_20190113_for_ogs.xlsx"
PROJECTS = ROOT / "projects"
RESULTS = ROOT / "results"
TEMP_ROOT = Path("/tmp/yangyi_wellhead_random_calibration")
DEFAULT_OGS = Path("/home/zhai/ogs-env/bin/ogs")
DEFAULT_SEED = 20260719

# Each active well is completed in the listed material. Bounds are deliberately
# targeted around the current values instead of randomizing unrelated media.
PARAMETERS = {
    5: (1.0e-10, 3.0e-9, "ZK403"),
    1: (3.0e-9, 3.0e-8, "ZK208"),
    12: (1.0e-12, 1.0e-11, "ZK203"),
}
BASELINE = {5: 1.0e-8, 1: 1.0e-8, 12: 2.5e-12}
WELL_WEIGHTS = {"ZK403": 0.60, "ZK208": 0.20, "ZK203": 0.20}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=24, help="number of random trials")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ogs", type=Path, default=DEFAULT_OGS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PLOT = load_module("yangyi_plot_wellhead", SHARED_DIR / "plot_wellhead_pressure.py")
COMPARE = load_module(
    "yangyi_compare_wellhead", SHARED_DIR / "compare_observed_wellhead_pressure.py"
)


def permeability_element(medium: ET.Element) -> ET.Element:
    for prop in medium.findall("./properties/property"):
        if prop.findtext("name") == "permeability":
            value = prop.find("value")
            if value is not None:
                return value
    raise RuntimeError(f"Medium {medium.attrib.get('id')} has no permeability value")


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(10.0 ** rng.uniform(math.log10(low), math.log10(high)))


def sample_trials(count: int, seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    trials: list[dict[str, object]] = [
        {
            "case_id": "baseline_000",
            "random_seed": seed,
            "is_baseline": True,
            **{f"k{mid}_m2": value for mid, value in BASELINE.items()},
        }
    ]
    for index in range(1, count + 1):
        trial: dict[str, object] = {
            "case_id": f"random_{index:03d}",
            "random_seed": seed,
            "is_baseline": False,
        }
        for material_id, (low, high, _) in PARAMETERS.items():
            trial[f"k{material_id}_m2"] = log_uniform(rng, low, high)
        trials.append(trial)
    return trials


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
        try:
            converted[key] = float(value)
        except (TypeError, ValueError):
            converted[key] = value
    return converted


def create_projects(trials: list[dict[str, object]]) -> None:
    PROJECTS.mkdir(parents=True, exist_ok=True)
    for trial in trials:
        tree = ET.parse(BASE_PROJECT)
        root = tree.getroot()
        media = {int(m.attrib["id"]): m for m in root.findall(".//medium")}
        for material_id in PARAMETERS:
            permeability_element(media[material_id]).text = (
                f"{float(trial[f'k{material_id}_m2']):.12e}"
            )
        prefix = root.find(".//time_loop/output/prefix")
        if prefix is None:
            raise RuntimeError("Base project has no output prefix")
        prefix.text = str(trial["case_id"])
        ET.indent(tree, space="  ")
        tree.write(
            PROJECTS / f"{trial['case_id']}.prj",
            encoding="UTF-8",
            xml_declaration=True,
        )


def score_summary(summary: list[dict[str, object]]) -> dict[str, float]:
    by_well = {str(row["well"]): row for row in summary}
    absolute = {
        well: float(by_well[well]["absolute_RMSE_MPa"]) for well in WELL_WEIGHTS
    }
    change = {
        well: float(by_well[well]["change_RMSE_MPa"]) for well in WELL_WEIGHTS
    }
    return {
        "objective_MPa": sum(WELL_WEIGHTS[w] * absolute[w] for w in WELL_WEIGHTS),
        "production_max_absolute_RMSE_MPa": max(absolute["ZK208"], absolute["ZK203"]),
        **{f"{well}_absolute_RMSE_MPa": absolute[well] for well in WELL_WEIGHTS},
        **{f"{well}_change_RMSE_MPa": change[well] for well in WELL_WEIGHTS},
    }


def run_trial(
    trial: dict[str, object],
    ogs: Path,
    observations,
    dates,
    resume: bool,
    force: bool,
) -> dict[str, object]:
    case_id = str(trial["case_id"])
    result_dir = RESULTS / case_id
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_dir / "summary.csv"
    if resume and not force and summary_path.exists():
        saved = read_one_row(summary_path)
        if saved.get("status") == "ok":
            return {**trial, **saved}

    temp_dir = TEMP_ROOT / case_id
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    started = time.monotonic()
    log_path = result_dir / "ogs.log"
    result: dict[str, object] = {}
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [
                    str(ogs),
                    str(PROJECTS / f"{case_id}.prj"),
                    "-m",
                    str(CASE_DIR),
                    "-o",
                    str(temp_dir),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"OGS returned {completed.returncode}; see {log_path}")

        wells = PLOT.read_wells(CASE_DIR)
        datasets = PLOT.read_pvd(temp_dir / f"{case_id}.pvd")
        simulated = PLOT.extract(datasets, wells, 1000.0, 9.81)
        simulation_csv = result_dir / "wellhead_pressure_timeseries.csv"
        PLOT.write_csv(simulation_csv, simulated)

        simulation_by_day = COMPARE.read_simulation(simulation_csv)
        comparison = COMPARE.build_comparison(simulation_by_day, observations, dates)
        metrics = COMPARE.metrics(comparison)
        COMPARE.write_rows(result_dir / "observed_vs_simulated.csv", comparison)
        COMPARE.write_rows(result_dir / "metrics_by_well.csv", metrics)
        result.update(score_summary(metrics))
        result.update({"status": "ok", "ogs_returncode": 0})
    except Exception as exc:
        result.update({"status": "failed", "error": str(exc)})
    finally:
        result["runtime_s"] = time.monotonic() - started
        shutil.rmtree(temp_dir, ignore_errors=True)

    write_csv(summary_path, [{**trial, **result}])
    return {**trial, **result}


def make_best_outputs(best: dict[str, object]) -> None:
    best_id = str(best["case_id"])
    source = RESULTS / best_id
    out_dir = ROOT / "best_case_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    simulation = COMPARE.read_simulation(source / "wellhead_pressure_timeseries.csv")
    observations, dates = COMPARE.read_observations(WORKBOOK)
    comparison = COMPARE.build_comparison(simulation, observations, dates)
    summary = COMPARE.metrics(comparison)
    COMPARE.write_rows(out_dir / "observed_vs_simulated_wellhead_pressure.csv", comparison)
    COMPARE.write_rows(out_dir / "comparison_metrics.csv", summary)
    COMPARE.make_plot(out_dir / "absolute_wellhead_pressure_comparison.png", comparison, False)
    COMPARE.make_plot(out_dir / "wellhead_pressure_change_comparison.png", comparison, True)
    COMPARE.make_completion_plot(
        out_dir / "depth_corrected_completion_pressure_comparison.png", comparison
    )
    COMPARE.make_combined_completion_plot(
        out_dir / "combined_six_line_completion_pressure_comparison.png", comparison
    )
    COMPARE.write_clean_workbook(out_dir / "clean_six_line_pressure_data.xlsx", comparison)
    shutil.copy2(source / "wellhead_pressure_timeseries.csv", out_dir)
    shutil.copy2(PROJECTS / f"{best_id}.prj", ROOT / "best_calibrated.prj")
    write_csv(ROOT / "best_case_parameters.csv", [best])


def main() -> None:
    args = parse_args()
    if args.cases < 1 or args.workers < 1:
        raise SystemExit("--cases and --workers must both be positive")
    if not args.ogs.is_file():
        raise SystemExit(f"OGS executable not found: {args.ogs}")
    PROJECTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    trials = sample_trials(args.cases, args.seed)
    write_csv(ROOT / "random_permeability_settings.csv", trials)
    create_projects(trials)
    observations, dates = COMPARE.read_observations(WORKBOOK)

    completed_rows: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_trial,
                trial,
                args.ogs.resolve(),
                observations,
                dates,
                args.resume,
                args.force,
            ): trial["case_id"]
            for trial in trials
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            row = future.result()
            completed_rows.append(row)
            print(
                f"[{index:02d}/{len(trials)}] {row['case_id']}: {row.get('status')} "
                f"objective={float(row.get('objective_MPa', math.nan)):.4f} MPa "
                f"runtime={float(row.get('runtime_s', math.nan)):.1f} s",
                flush=True,
            )
            write_csv(ROOT / "ranking_partial.csv", completed_rows)

    successful = [row for row in completed_rows if row.get("status") == "ok"]
    failed = [row for row in completed_rows if row.get("status") != "ok"]
    successful.sort(key=lambda row: float(row["objective_MPa"]))
    ranked: list[dict[str, object]] = []
    for rank, row in enumerate(successful, start=1):
        ranked.append({"rank": rank, **row})
    ranked.extend({"rank": "", **row} for row in failed)
    write_csv(ROOT / "ranking_all_cases.csv", ranked)
    write_csv(ROOT / "top_10_cases.csv", ranked[:10])
    if not successful:
        raise SystemExit("No trial completed successfully")
    make_best_outputs(successful[0])

    print("\nBest trials:")
    for row in successful[:10]:
        print(
            f"{row['case_id']}: objective={float(row['objective_MPa']):.4f}, "
            f"k5={float(row['k5_m2']):.3e}, k1={float(row['k1_m2']):.3e}, "
            f"k12={float(row['k12_m2']):.3e}, "
            f"RMSE(ZK403/ZK208/ZK203)="
            f"{float(row['ZK403_absolute_RMSE_MPa']):.3f}/"
            f"{float(row['ZK208_absolute_RMSE_MPa']):.3f}/"
            f"{float(row['ZK203_absolute_RMSE_MPa']):.3f} MPa"
        )


if __name__ == "__main__":
    main()
