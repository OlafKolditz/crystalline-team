#!/usr/bin/env python3
"""Plot simulated and monitored active-well pressure for all four models."""

from __future__ import annotations

import argparse
import base64
import csv
import math
import re
import struct
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pyvista as pv


HERE = Path(__file__).resolve().parent
SECONDS_PER_DAY = 86_400.0
RHO = 1000.0
GRAVITY = 9.81
COLORS = {"ZK403": "#2ca02c", "ZK208": "#d62728", "ZK203": "#ff7f0e"}
ROLES = {"ZK403": "Injection", "ZK208": "Production", "ZK203": "Production"}
OBSERVED_COLUMNS = {
    "ZK403": "Reinj_Pout_MPa_avg",
    "ZK208": "ZK208_P_MPa_avg",
    "ZK203": "ZK203_P_MPa_avg",
}
FULL_REFERENCE_SOURCES = (
    HERE.parent / "case1_k9_2e-11_k10_4e-13" / "source_terms"
)
WORKBOOK = HERE / "monitoring_data" / "yangyi_after_20190113_for_ogs.xlsx"
MODELS = {
    "DFNM": {
        "pvd": HERE / "DFNM" / "output" / "bestfit_DFNM_200d.pvd",
        "wells": ("ZK403", "ZK208", "ZK203"),
        "sources": HERE / "DFNM" / "source_terms",
        "source_pattern": "{well}_single_source_sink_bulk_node.vtu",
    },
    "DFNM-LRZ": {
        "pvd": HERE / "DFNM-LRZ" / "output" / "bestfit_DFNM_LRZ_200d.pvd",
        "wells": ("ZK403", "ZK208", "ZK203"),
        "sources": HERE / "DFNM-LRZ" / "source_terms",
        "source_pattern": "{well}_single_source_sink_bulk_node.vtu",
    },
    "DFN": {
        "pvd": HERE / "DFN" / "output" / "bestfit_DFN_200d.pvd",
        "wells": ("ZK403", "ZK208"),
        "sources": HERE / "DFN",
        "source_pattern": "{well}_2d_source.vtu",
    },
    "DFN-LRZ": {
        "pvd": HERE / "DFN-LRZ" / "output" / "bestfit_DFNM_LRZ_200d.pvd",
        "wells": ("ZK403", "ZK208", "ZK203"),
        "sources": HERE / "DFN-LRZ" / "source_terms",
        "source_pattern": "{well}_single_source_sink_bulk_node.vtu",
    },
}
MODEL_ORDER = ("DFNM", "DFNM-LRZ", "DFN", "DFN-LRZ")


@dataclass(frozen=True)
class Well:
    name: str
    node_id: int
    depth_m: float


def point_scalar(path: Path, name: str) -> float:
    mesh = pv.read(path)
    if name not in mesh.point_data:
        raise RuntimeError(f"{name} is absent from {path}")
    return float(np.asarray(mesh.point_data[name]).reshape(-1)[0])


def wells_for_model(settings: dict[str, object]) -> list[Well]:
    wells = []
    for name in settings["wells"]:
        source = Path(settings["sources"]) / str(settings["source_pattern"]).format(well=name)
        reference = FULL_REFERENCE_SOURCES / f"{name}_single_source_sink_bulk_node.vtu"
        wells.append(
            Well(
                name=name,
                node_id=int(point_scalar(source, "bulk_node_ids")),
                depth_m=point_scalar(reference, "depth_m"),
            )
        )
    return wells


def pvd_series(path: Path) -> list[tuple[float, Path]]:
    root = ET.parse(path).getroot()
    series = [
        (float(item.attrib["timestep"]), path.parent / item.attrib["file"])
        for item in root.iter("DataSet")
    ]
    if not series:
        raise RuntimeError(f"No datasets in {path}")
    return sorted(series)


def decode_pressure(path: Path) -> np.ndarray:
    """Decode only OGS' zlib-compressed appended pressure array."""
    with path.open("rb") as stream:
        header = stream.read(65_536)
        appended = header.find(b"<AppendedData")
        marker = header.find(b"_", appended)
        match = re.search(
            rb'<DataArray[^>]*Name="pressure"[^>]*offset="(\d+)"[^>]*/>',
            header[:appended],
        )
        if appended < 0 or marker < 0 or match is None:
            raise RuntimeError(f"Cannot locate appended pressure in {path}")
        offset = int(match.group(1))
        stream.seek(marker + 1 + offset)
        first = base64.b64decode(stream.read(32))
        blocks, block_size, final_size = struct.unpack("<3Q", first)
        header_bytes = 8 * (3 + blocks)
        header_chars = 4 * ((header_bytes + 2) // 3)
        stream.seek(marker + 1 + offset)
        compressed_sizes = struct.unpack(
            "<" + "Q" * (3 + blocks), base64.b64decode(stream.read(header_chars))
        )[3:]
        compressed_bytes = sum(compressed_sizes)
        encoded_chars = 4 * ((compressed_bytes + 2) // 3)
        compressed = base64.b64decode(stream.read(encoded_chars))
    decoded = []
    start = 0
    for size in compressed_sizes:
        decoded.append(zlib.decompress(compressed[start : start + size]))
        start += size
    pressure = np.frombuffer(b"".join(decoded), dtype="<f8")
    expected = (blocks - 1) * block_size + final_size
    if pressure.nbytes != expected:
        raise RuntimeError(f"Pressure byte count mismatch in {path}")
    return pressure


def extract_model(model: str, settings: dict[str, object]) -> pd.DataFrame:
    wells = wells_for_model(settings)
    rows = []
    for seconds, path in pvd_series(Path(settings["pvd"])):
        pressure = decode_pressure(path)
        for well in wells:
            node_mpa = float(pressure[well.node_id]) / 1.0e6
            correction_mpa = RHO * GRAVITY * well.depth_m / 1.0e6
            rows.append(
                {
                    "model": model,
                    "time_days": seconds / SECONDS_PER_DAY,
                    "well": well.name,
                    "role": ROLES[well.name],
                    "node_id": well.node_id,
                    "completion_depth_m": well.depth_m,
                    "completion_pressure_MPa": node_mpa,
                    "hydrostatic_depth_correction_MPa": correction_mpa,
                    "simulated_wellhead_pressure_MPa": node_mpa - correction_mpa,
                }
            )
    return pd.DataFrame(rows)


def observations() -> pd.DataFrame:
    raw = pd.read_excel(WORKBOOK, sheet_name="Daily_ops_after20190113")
    rows = []
    for well, column in OBSERVED_COLUMNS.items():
        for _, item in raw.iterrows():
            day = pd.to_numeric(item.get("time_days_from_20190113"), errors="coerce")
            pressure = pd.to_numeric(item.get(column), errors="coerce")
            if pd.notna(day) and pd.notna(pressure):
                rows.append(
                    {
                        "time_days": float(day),
                        "well": well,
                        "monitoring_wellhead_pressure_MPa": float(pressure),
                        "observed_source_column": column,
                    }
                )
    return pd.DataFrame(rows)


def plot_panel(
    path: Path,
    model: str,
    simulation: pd.DataFrame,
    observed: pd.DataFrame,
    wells: list[str],
    completion_depth: bool = False,
) -> None:
    fig, axes = plt.subplots(
        len(wells), 1, figsize=(11.0, max(5.0, 4.2 * len(wells))), sharex=True
    )
    axes = np.atleast_1d(axes)
    for axis, well in zip(axes, wells):
        sim = simulation[simulation["well"] == well].sort_values("time_days")
        obs = observed[observed["well"] == well].sort_values("time_days")
        color = COLORS[well]
        depth_m = float(sim["completion_depth_m"].iloc[0])
        correction_mpa = float(sim["hydrostatic_depth_correction_MPa"].iloc[0])
        if completion_depth:
            simulated = sim["completion_pressure_MPa"]
            monitored = obs["monitoring_wellhead_pressure_MPa"] + correction_mpa
            ylabel = "Pressure at completion depth (MPa)"
            depth_note = f", depth {depth_m:.2f} m, +{correction_mpa:.3f} MPa hydrostatic"
        else:
            simulated = sim["simulated_wellhead_pressure_MPa"]
            monitored = obs["monitoring_wellhead_pressure_MPa"]
            ylabel = "Wellhead pressure (MPa)"
            depth_note = f", depth correction −{correction_mpa:.3f} MPa"
        axis.plot(
            sim["time_days"],
            simulated,
            color=color,
            linestyle="--",
            linewidth=2.2,
            label="Simulation",
        )
        axis.plot(
            obs["time_days"],
            monitored,
            color=color,
            linestyle="-",
            marker="o",
            markersize=3.2,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.8,
            linewidth=1.2,
            label="Monitoring data",
        )
        axis.set_ylabel(ylabel)
        axis.set_title(f"{well} — {ROLES[well].lower()}{depth_note}", loc="left")
        axis.grid(True, color="0.87", linewidth=0.8)
        axis.legend(frameon=False, loc="best")
    axes[-1].set_xlabel("Time since 2019-01-13 (days)")
    axes[-1].set_xlim(left=0)
    group = (
        "Injection well"
        if wells == ["ZK403"]
        else ("Production well" if len(wells) == 1 else "Production wells")
    )
    reference = "completion-depth" if completion_depth else "wellhead"
    fig.suptitle(
        f"{model}: {group} {reference} pressure — simulation vs monitoring", fontsize=14
    )
    fig.tight_layout(rect=(0.02, 0.01, 0.99, 0.95))
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def comparison_and_metrics(simulation: pd.DataFrame, observed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sim = simulation.copy()
    sim["day_key"] = sim["time_days"].round().astype(int)
    obs = observed.copy()
    obs["day_key"] = obs["time_days"].round().astype(int)
    merged = sim.merge(
        obs[["well", "day_key", "monitoring_wellhead_pressure_MPa", "observed_source_column"]],
        on=["well", "day_key"], how="left",
    )
    merged["residual_sim_minus_monitoring_MPa"] = (
        merged["simulated_wellhead_pressure_MPa"]
        - merged["monitoring_wellhead_pressure_MPa"]
    )
    merged["monitoring_completion_pressure_MPa"] = (
        merged["monitoring_wellhead_pressure_MPa"]
        + merged["hydrostatic_depth_correction_MPa"]
    )
    valid = merged.dropna(subset=["monitoring_wellhead_pressure_MPa"]).copy()
    metrics = []
    for (model, well), group in valid.groupby(["model", "well"]):
        residual = group["residual_sim_minus_monitoring_MPa"].to_numpy(float)
        metrics.append(
            {
                "model": model,
                "well": well,
                "n": len(group),
                "bias_MPa": float(residual.mean()),
                "MAE_MPa": float(np.abs(residual).mean()),
                "RMSE_MPa": float(np.sqrt(np.mean(residual**2))),
            }
        )
    return merged, pd.DataFrame(metrics)


def plot_all_wells_one_axis(
    path: Path,
    model: str,
    simulation: pd.DataFrame,
    observed: pd.DataFrame,
    metrics: pd.DataFrame,
    wells: list[str],
    completion_depth: bool = False,
) -> None:
    """Overlay all active wells for one case in a single coordinate system."""
    fig, axis = plt.subplots(figsize=(15, 8.5))
    plotted_values = []
    for well in wells:
        sim = simulation[simulation["well"] == well].sort_values("time_days")
        obs = observed[observed["well"] == well].sort_values("time_days")
        color = COLORS[well]
        correction_mpa = float(sim["hydrostatic_depth_correction_MPa"].iloc[0])
        if completion_depth:
            simulated = sim["completion_pressure_MPa"]
            monitored = obs["monitoring_wellhead_pressure_MPa"] + correction_mpa
        else:
            simulated = sim["simulated_wellhead_pressure_MPa"]
            monitored = obs["monitoring_wellhead_pressure_MPa"]
        plotted_values.extend(np.asarray(simulated, dtype=float))
        plotted_values.extend(np.asarray(monitored, dtype=float))
        axis.plot(
            sim["time_days"],
            simulated,
            "--",
            color=color,
            linewidth=2.2,
        )
        axis.plot(
            obs["time_days"],
            monitored,
            "-o",
            color=color,
            markerfacecolor="white",
            markersize=4,
            linewidth=1.4,
        )
    values = np.asarray(plotted_values, dtype=float)
    finite = values[np.isfinite(values)]
    scale_note = ""
    if finite.size and np.nanmax(np.abs(finite)) > 100.0:
        axis.set_yscale("symlog", linthresh=0.1)
        scale_note = "; symmetric-log pressure scale"
    reference = "completion-depth" if completion_depth else "wellhead"
    axis.set_title(
        f"{model}: all active-well {reference} pressures in one coordinate system{scale_note}"
    )
    axis.set_xlabel("Time since 2019-01-13 (days)")
    axis.set_ylabel(
        "Pressure at completion depth (MPa)"
        if completion_depth
        else "Wellhead pressure (MPa)"
    )
    axis.set_xlim(left=0)
    axis.grid(True, which="both", color="0.87", linewidth=0.8)
    well_handles = []
    for well in wells:
        rmse = float(metrics.loc[metrics["well"] == well, "RMSE_MPa"].iloc[0])
        well_handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS[well],
                linewidth=2,
                label=f"{well} — {ROLES[well]} (RMSE {rmse:.3f} MPa)",
            )
        )
    first_legend = axis.legend(
        handles=well_handles,
        title="Well color and RMSE",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )
    axis.add_artist(first_legend)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="-",
                marker="o",
                markerfacecolor="white",
                label="Monitoring data",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=2,
                label="Simulation",
            ),
        ],
        title="Line style",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 0.76, 1))
    fig.savefig(path, dpi=250)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODEL_ORDER, "all"], default="all")
    args = parser.parse_args()
    selected = (
        {name: MODELS[name] for name in MODEL_ORDER}
        if args.model == "all"
        else {args.model: MODELS[args.model]}
    )
    observed = observations()
    all_simulation = []
    all_comparison = []
    all_metrics = []
    for model, settings in selected.items():
        pvd = Path(settings["pvd"])
        if not pvd.exists():
            raise FileNotFoundError(f"Missing completed output {pvd}")
        output = HERE / model / "pressure_comparison"
        output.mkdir(exist_ok=True)
        simulation = extract_model(model, settings)
        comparison, metrics = comparison_and_metrics(simulation, observed)
        simulation.to_csv(output / "simulated_wellhead_pressure.csv", index=False)
        comparison.to_csv(output / "simulation_vs_monitoring_pressure.csv", index=False)
        metrics.to_csv(output / "pressure_error_metrics.csv", index=False)
        plot_panel(
            output / "injection_well_pressure.png", model, simulation, observed, ["ZK403"]
        )
        production = [well for well in ("ZK208", "ZK203") if well in settings["wells"]]
        plot_panel(
            output / "production_well_pressure.png", model, simulation, observed, production
        )
        plot_panel(
            output / "injection_completion_pressure.png",
            model,
            simulation,
            observed,
            ["ZK403"],
            completion_depth=True,
        )
        plot_panel(
            output / "production_completion_pressure.png",
            model,
            simulation,
            observed,
            production,
            completion_depth=True,
        )
        active_wells = list(settings["wells"])
        plot_all_wells_one_axis(
            output / "all_wells_wellhead_pressure_one_coordinate.png",
            model,
            simulation,
            observed,
            metrics,
            active_wells,
        )
        plot_all_wells_one_axis(
            output / "all_wells_completion_pressure_one_coordinate.png",
            model,
            simulation,
            observed,
            metrics,
            active_wells,
            completion_depth=True,
        )
        all_simulation.append(simulation)
        all_comparison.append(comparison)
        all_metrics.append(metrics)
        print(f"Created pressure comparisons for {model}: {', '.join(settings['wells'])}")
    if args.model == "all":
        pd.concat(all_simulation, ignore_index=True).to_csv(
            HERE / "all_models_simulated_wellhead_pressure.csv", index=False
        )
        pd.concat(all_comparison, ignore_index=True).to_csv(
            HERE / "all_models_simulation_vs_monitoring_pressure.csv", index=False
        )
        pd.concat(all_metrics, ignore_index=True).to_csv(
            HERE / "all_models_pressure_error_metrics.csv", index=False
        )


if __name__ == "__main__":
    main()
