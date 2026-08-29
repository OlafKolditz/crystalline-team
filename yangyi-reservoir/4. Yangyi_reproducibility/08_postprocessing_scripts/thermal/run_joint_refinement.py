#!/usr/bin/env python3
"""Local M203 deep-temperature/thermal-dispersivity refinement for four years."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

CASE_DIR = Path(__file__).resolve().parent
PARENT = CASE_DIR.parent / "case3_split_storage_heat_exchange_4y"
sys.path.insert(0, str(PARENT / ".python_deps"))
os.environ.setdefault("MPLCONFIGDIR", str(CASE_DIR / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv


SOURCE_NAME = "M203T194C_S203A0p3x_S208A6x_alphaL30m"
SOURCE_DIR = PARENT / "dispersion_runs" / SOURCE_NAME
SOURCE_PROJECT = SOURCE_DIR / f"{SOURCE_NAME}.prj"
INPUT_MESH = PARENT / "input_mesh"
EXCEL = PARENT / "Power output and PT.xls"
OGS = Path(os.environ.get("OGS_BIN", "/home/zhai/ogs-env/bin/ogs"))
WORKERS = int(os.environ.get("SWEEP_WORKERS", "3"))

START_DATE = pd.Timestamp("2019-01-01")
END_TIME_SECONDS = 126230400.0
M203_TEMPERATURES_C = (190.0, 192.0, 194.0)
THERMAL_DISPERSIVITIES_M = (25.0, 30.0, 35.0)
WELLS = ("ZK203", "ZK208")
STAGES = {
    "initial": (12.0, 120.0),
    "front": (120.0, 240.0),
    "long_term": (240.0, 1461.0),
}


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def case_name(temperature_c: float, alpha_m: float) -> str:
    return f"M203T{tag(temperature_c)}C_S203A0p3x_S208A6x_alphaL{tag(alpha_m)}m"


def load_observed() -> pd.DataFrame:
    observed = pd.read_excel(
        EXCEL,
        sheet_name="Sheet1",
        header=None,
        skiprows=3,
        usecols=[0, 2, 4],
        names=["date", "ZK203_observed_C", "ZK208_observed_C"],
    )
    observed["date"] = pd.to_datetime(observed["date"], errors="coerce")
    observed["time_days"] = (observed["date"] - START_DATE).dt.total_seconds() / 86400.0
    for well in WELLS:
        column = f"{well}_observed_C"
        observed[column] = pd.to_numeric(observed[column], errors="coerce")
        observed.loc[~observed[column].between(0.0, 250.0), column] = np.nan
    return observed.loc[observed["time_days"].between(12.0, 1461.0)].copy()


OBSERVED = load_observed()


def pvd_entries(pvd: Path) -> list[tuple[float, Path]]:
    root = ET.parse(pvd).getroot()
    return [
        (float(item.attrib["timestep"]), pvd.parent / item.attrib["file"])
        for item in root.findall("./Collection/DataSet")
    ]


def prepare_case(temperature_c: float, alpha_m: float) -> tuple[str, Path, Path]:
    name = case_name(temperature_c, alpha_m)
    run_dir = CASE_DIR / "runs" / name

    # Reuse the already completed centre point and retain a self-contained copy.
    if name == SOURCE_NAME and not run_dir.exists():
        shutil.copytree(SOURCE_DIR, run_dir)

    output_dir = run_dir / "output"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    local_mesh = run_dir / "input_mesh"
    if not local_mesh.exists():
        shutil.copytree(INPUT_MESH, local_mesh)

    tree = ET.parse(SOURCE_PROJECT)
    root = tree.getroot()
    for mesh in root.findall("./meshes/mesh"):
        mesh.text = f"input_mesh/{Path(mesh.text).name}"

    deep_parameter = next(
        parameter
        for parameter in root.findall("./parameters/parameter")
        if parameter.findtext("name") == "M203_deep_return_temperature"
    )
    deep_parameter.find("value").text = f"{temperature_c + 273.15:g}"

    changed = 0
    for medium in root.findall("./media/medium"):
        for prop in medium.findall("./properties/property"):
            if prop.findtext("name") == "thermal_longitudinal_dispersivity":
                prop.find("value").text = f"{alpha_m:g}"
                changed += 1
    if changed == 0:
        raise ValueError("No thermal_longitudinal_dispersivity properties found")

    root.find("./time_loop/output/prefix").text = name
    project = run_dir / f"{name}.prj"
    ET.indent(tree, space="  ")
    tree.write(project, encoding="UTF-8", xml_declaration=True)
    return name, run_dir, project


def extract_temperature(name: str, run_dir: Path) -> pd.DataFrame:
    coordinates = {
        well: pv.read(run_dir / "input_mesh" / f"{well}.vtu").points[0]
        for well in WELLS
    }
    records = []
    for seconds, vtu in pvd_entries(run_dir / "output" / f"{name}.pvd"):
        mesh = pv.read(vtu)
        temperature = np.asarray(mesh.point_data["temperature"], dtype=float) - 273.15
        row = {
            "time_seconds": seconds,
            "time_days": seconds / 86400.0,
            "date": START_DATE + pd.to_timedelta(seconds, unit="s"),
            "domain_min_temperature_C": float(temperature.min()),
            "domain_max_temperature_C": float(temperature.max()),
        }
        for well, coordinate in coordinates.items():
            row[f"{well}_temperature_C"] = float(
                temperature[mesh.find_closest_point(coordinate)]
            )
        records.append(row)
    result = pd.DataFrame(records)
    result.to_csv(
        run_dir / "production_temperature_4y.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    return result


def residuals(simulated: pd.DataFrame, well: str, start: float, end: float) -> np.ndarray:
    column = f"{well}_observed_C"
    valid = OBSERVED.dropna(subset=[column])
    valid = valid.loc[valid["time_days"].between(start, end, inclusive="left")]
    predicted = np.interp(
        valid["time_days"], simulated["time_days"], simulated[f"{well}_temperature_C"]
    )
    return predicted - valid[column].to_numpy()


def calculate_metrics(
    name: str, temperature_c: float, alpha_m: float, simulated: pd.DataFrame
) -> dict[str, float]:
    row: dict[str, float] = {
        "case": name,
        "M203_deep_temperature_C": temperature_c,
        "thermal_longitudinal_dispersivity_m": alpha_m,
        "outputs": len(simulated),
        "ZK203_final_temperature_C": float(simulated.iloc[-1]["ZK203_temperature_C"]),
        "ZK208_final_temperature_C": float(simulated.iloc[-1]["ZK208_temperature_C"]),
    }
    full_rmses = []
    stage_rmses = []
    for well in WELLS:
        full = residuals(simulated, well, 12.0, 1461.0001)
        row[f"{well}_bias_C"] = float(full.mean())
        row[f"{well}_RMSE_C"] = float(np.sqrt(np.mean(full**2)))
        full_rmses.append(row[f"{well}_RMSE_C"])
        for stage, (start, end) in STAGES.items():
            values = residuals(simulated, well, start, end)
            metric = float(np.sqrt(np.mean(values**2)))
            row[f"{well}_{stage}_RMSE_C"] = metric
            stage_rmses.append(metric)
    row["full_mean_RMSE_C"] = float(np.mean(full_rmses))
    row["stage_balanced_RMSE_C"] = float(np.mean(stage_rmses))
    return row


def run_case(temperature_c: float, alpha_m: float) -> dict[str, float]:
    name, run_dir, project = prepare_case(temperature_c, alpha_m)
    csv_path = run_dir / "production_temperature_4y.csv"
    pvd_path = run_dir / "output" / f"{name}.pvd"
    if csv_path.exists() and pvd_path.exists():
        entries = pvd_entries(pvd_path)
        if entries and abs(entries[-1][0] - END_TIME_SECONDS) < 1.0:
            print(f"REUSE {name}", flush=True)
            simulated = pd.read_csv(csv_path)
            return calculate_metrics(name, temperature_c, alpha_m, simulated)

    print(f"START {name}", flush=True)
    started = time.monotonic()
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    log_path = run_dir / "output" / "ogs.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [str(OGS), "-l", "warn", project.name, "-o", "output"],
            cwd=run_dir,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed; see {log_path}")
    simulated = extract_temperature(name, run_dir)
    row = calculate_metrics(name, temperature_c, alpha_m, simulated)
    row["runtime_seconds"] = time.monotonic() - started
    print(
        f"DONE  {name}: full={row['full_mean_RMSE_C']:.3f} C, "
        f"stage-balanced={row['stage_balanced_RMSE_C']:.3f} C",
        flush=True,
    )
    return row


def make_plots(ranking: pd.DataFrame) -> None:
    colors = {190.0: "#2878b5", 192.0: "#f28522", 194.0: "#c82423"}
    styles = {25.0: ":", 30.0: "-", 35.0: "--"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    for axis, well in zip(axes, WELLS):
        axis.scatter(
            OBSERVED["date"], OBSERVED[f"{well}_observed_C"],
            s=7, color="0.2", alpha=0.4, label="Observed",
        )
        for temperature_c in M203_TEMPERATURES_C:
            for alpha_m in THERMAL_DISPERSIVITIES_M:
                name = case_name(temperature_c, alpha_m)
                curve = pd.read_csv(
                    CASE_DIR / "runs" / name / "production_temperature_4y.csv",
                    parse_dates=["date"],
                )
                axis.plot(
                    curve["date"], curve[f"{well}_temperature_C"],
                    color=colors[temperature_c], ls=styles[alpha_m], lw=1.35,
                    label=f"M203={temperature_c:g} C, alphaL={alpha_m:g} m",
                )
        axis.set_title(well, loc="left", fontweight="bold")
        axis.set_ylabel("Production temperature (degC)")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Date")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.83, 0.5), fontsize=8)
    fig.suptitle("Local joint refinement: M203 deep temperature and thermal dispersivity")
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(CASE_DIR / "all_joint_cases_comparison_4y.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    best = ranking.iloc[0]
    curve = pd.read_csv(
        CASE_DIR / "runs" / best["case"] / "production_temperature_4y.csv",
        parse_dates=["date"],
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    for axis, well in zip(axes, WELLS):
        axis.scatter(
            OBSERVED["date"], OBSERVED[f"{well}_observed_C"],
            s=8, color="0.2", alpha=0.45, label="Observed",
        )
        axis.plot(
            curve["date"], curve[f"{well}_temperature_C"],
            color="#1f5aa6", lw=2.0,
            label=(f"Best: M203={best['M203_deep_temperature_C']:g} C, "
                   f"alphaL={best['thermal_longitudinal_dispersivity_m']:g} m"),
        )
        axis.set_title(well, loc="left", fontweight="bold")
        axis.set_ylabel("Production temperature (degC)")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("Date")
    fig.suptitle("Best stage-balanced local-refinement case")
    fig.tight_layout()
    fig.savefig(CASE_DIR / "best_joint_case_comparison_4y.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    designs = [
        (temperature_c, alpha_m)
        for temperature_c in M203_TEMPERATURES_C
        for alpha_m in THERMAL_DISPERSIVITIES_M
    ]
    rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_case, *design): design for design in designs}
        for future in as_completed(futures):
            rows.append(future.result())

    ranking = pd.DataFrame(rows).sort_values(
        ["stage_balanced_RMSE_C", "full_mean_RMSE_C"]
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(CASE_DIR / "joint_refinement_ranking.csv", index=False)
    make_plots(ranking)

    columns = [
        "rank", "M203_deep_temperature_C", "thermal_longitudinal_dispersivity_m",
        "ZK203_initial_RMSE_C", "ZK203_front_RMSE_C", "ZK203_long_term_RMSE_C",
        "ZK208_initial_RMSE_C", "ZK208_front_RMSE_C", "ZK208_long_term_RMSE_C",
        "full_mean_RMSE_C", "stage_balanced_RMSE_C",
    ]
    print("\nJOINT REFINEMENT RANKING", flush=True)
    print(ranking[columns].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
