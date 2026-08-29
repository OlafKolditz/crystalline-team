#!/usr/bin/env python3
"""Compare simulated and observed monitoring-well hydraulic-head changes."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pyvista as pv


HERE = Path(__file__).resolve().parent
RHO = 1000.0
GRAVITY = 9.81
SECONDS_PER_DAY = 86_400.0
WELLS = ("SC211", "ZK204", "ZK206", "ZK207", "ZK401", "ZK402", "ZK501")
COLORS = {
    "SC211": "#9467bd",
    "ZK204": "#1f77b4",
    "ZK206": "#8c564b",
    "ZK207": "#17becf",
    "ZK401": "#2ca02c",
    "ZK402": "#d62728",
    "ZK501": "#ff7f0e",
}
MODEL_STYLES = {
    "DFNM": (0, (8, 3)),
    "DFNM-LRZ": (0, (5, 2)),
    "DFN": (0, (1.5, 2)),
    "DFN-LRZ": (0, (3, 2)),
}
MODEL_ALPHA = {
    "DFNM": 1.0,
    "DFNM-LRZ": 0.8,
    "DFN": 0.5,
    "DFN-LRZ": 0.65,
}
OBS_COLUMNS = {
    well: f"{well}_obs_delta_h_m_from_filtered_baseline" for well in WELLS
}
FULL_MONITOR_CSV = (
    HERE.parent
    / "case1_k9_2e-11_k10_4e-13"
    / "monitoring"
    / "recommended_monitoring_points_all7_ZK501_bedrock.csv"
)
NOMATRIX_MONITOR_CSV = (
    HERE.parent
    / "case1_k9_2e-11_k10_4e-13_noMatrix"
    / "monitoring"
    / "recommended_monitoring_points_all7_ZK501_bedrock_noMatrix.csv"
)
WORKBOOK = HERE / "monitoring_data" / "yangyi_after_20190113_for_ogs.xlsx"
MODELS = {
    "DFNM": {
        "pvd": HERE / "DFNM" / "output" / "bestfit_DFNM_200d.pvd",
        "bulk": HERE / "DFNM" / "input_mesh" / "reservoir_without_lrz.vtu",
        "mapping": "full",
    },
    "DFNM-LRZ": {
        "pvd": HERE / "DFNM-LRZ" / "output" / "bestfit_DFNM_LRZ_200d.pvd",
        "bulk": HERE / "DFNM-LRZ" / "input_mesh" / "reservoir_with_lowres.vtu",
        "mapping": "full",
        "folder": "DFNM-LRZ",
    },
    "DFN": {
        "pvd": HERE / "DFN" / "output" / "bestfit_DFN_200d.pvd",
        "bulk": HERE / "DFN" / "yangyi_2d_fractures.vtu",
        "mapping": "nearest",
    },
    "DFN-LRZ": {
        "pvd": HERE / "DFN-LRZ" / "output" / "bestfit_DFNM_LRZ_200d.pvd",
        "bulk": HERE / "DFN-LRZ" / "input_mesh" / "reservoir_with_lowres.vtu",
        "mapping": "nomatrix",
        "folder": "DFN-LRZ",
    },
}
MODEL_ORDER = ("DFNM", "DFNM-LRZ", "DFN", "DFN-LRZ")


def observations() -> pd.DataFrame:
    frame = pd.read_excel(WORKBOOK, sheet_name="Monitor_compare_after20190113")
    frame = frame.rename(columns={"time_days_from_20190113": "time_days"})
    return frame[["Date", "time_days", *OBS_COLUMNS.values()]].copy()


def pvd_series(path: Path) -> list[tuple[float, Path]]:
    root = ET.parse(path).getroot()
    rows = []
    for dataset in root.findall(".//DataSet"):
        rows.append(
            (
                float(dataset.attrib["timestep"]) / SECONDS_PER_DAY,
                path.parent / dataset.attrib["file"],
            )
        )
    rows.sort(key=lambda item: item[0])
    if not rows or any(not file.exists() for _, file in rows):
        raise RuntimeError(f"Incomplete PVD series: {path}")
    return rows


def monitoring_mapping(model: str, settings: dict[str, object]) -> pd.DataFrame:
    if settings["mapping"] == "full":
        frame = pd.read_csv(FULL_MONITOR_CSV)
        frame["mapping_distance_m"] = frame["nearest_node_distance_m"]
    elif settings["mapping"] == "nomatrix":
        frame = pd.read_csv(NOMATRIX_MONITOR_CSV)
        frame["mapping_distance_m"] = frame["noMatrix_node_distance_from_target_m"]
    else:
        frame = pd.read_csv(FULL_MONITOR_CSV)
        # A fracture-only model can represent only monitoring points located on
        # explicit fault/fracture materials.  Do not project LRZ (12) or matrix
        # (0) observations onto an unrelated nearest fracture.
        frame = frame[~frame["MaterialID"].isin([0, 12])].copy()
        bulk = pv.read(settings["bulk"])
        node_ids = []
        distances = []
        for xyz in frame[["x", "y", "z"]].to_numpy(float):
            node_id = int(bulk.find_closest_point(xyz))
            node_ids.append(node_id)
            distances.append(float(np.linalg.norm(bulk.points[node_id] - xyz)))
        frame["bulk_node_id"] = node_ids
        frame["mapping_distance_m"] = distances
    frame = frame.copy()
    frame["model"] = model
    frame["point_index"] = frame.groupby("well").cumcount() + 1
    frame["point_label"] = (
        frame["well"].astype(str)
        + "_"
        + frame["recommended_type"].astype(str)
        + "_"
        + frame["point_index"].astype(str)
    )
    return frame


def extract_model(model: str, settings: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = monitoring_mapping(model, settings)
    node_ids = mapping["bulk_node_id"].astype(int).to_numpy()
    series = pvd_series(Path(settings["pvd"]))
    records = []
    for time_days, path in series:
        mesh = pv.read(path)
        pressure = np.asarray(mesh.point_data["pressure"]).reshape(-1)
        for row, node_id in zip(mapping.itertuples(index=False), node_ids):
            z = float(mesh.points[node_id, 2])
            records.append(
                {
                    "model": model,
                    "time_days": time_days,
                    "well": row.well,
                    "point_label": row.point_label,
                    "bulk_node_id": node_id,
                    "pressure_Pa": float(pressure[node_id]),
                    "head_m": float(pressure[node_id]) / (RHO * GRAVITY) + z,
                }
            )
    simulation = pd.DataFrame(records)
    zero_day = 1.0
    pieces = []
    for _, group in simulation.groupby("point_label", sort=False):
        group = group.sort_values("time_days").copy()
        h0 = float(np.interp(zero_day, group["time_days"], group["head_m"]))
        group["sim_delta_head_m"] = group["head_m"] - h0
        pieces.append(group)
    simulation = pd.concat(pieces, ignore_index=True)
    mean_simulation = (
        simulation.groupby(["model", "time_days", "well"], as_index=False)
        .agg(sim_delta_head_m=("sim_delta_head_m", "mean"), n_points=("point_label", "nunique"))
    )
    return mean_simulation, mapping


def compare(simulation: pd.DataFrame, observed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    model = str(simulation["model"].iloc[0])
    available_wells = [well for well in WELLS if well in set(simulation["well"])]
    for well in available_wells:
        sim = simulation[simulation["well"] == well].sort_values("time_days")
        column = OBS_COLUMNS[well]
        obs = observed[["Date", "time_days", column]].dropna().rename(columns={column: "observed_delta_head_m"})
        interpolated = np.interp(obs["time_days"], sim["time_days"], sim["sim_delta_head_m"])
        for item, value in zip(obs.itertuples(index=False), interpolated):
            rows.append(
                {
                    "model": model,
                    "well": well,
                    "Date": item.Date,
                    "time_days": float(item.time_days),
                    "observed_delta_head_m": float(item.observed_delta_head_m),
                    "simulated_delta_head_m": float(value),
                    "residual_sim_minus_observed_m": float(value - item.observed_delta_head_m),
                }
            )
    comparison = pd.DataFrame(rows)
    metrics = []
    for well, group in comparison.groupby("well", sort=False):
        residual = group["residual_sim_minus_observed_m"].to_numpy(float)
        metrics.append(
            {
                "model": model,
                "well": well,
                "n": len(group),
                "bias_m": float(residual.mean()),
                "MAE_m": float(np.abs(residual).mean()),
                "RMSE_m": float(np.sqrt(np.mean(residual**2))),
            }
        )
    return comparison, pd.DataFrame(metrics)


def plot_model(model: str, simulation: pd.DataFrame, observed: pd.DataFrame, metrics: pd.DataFrame, out: Path) -> None:
    available_wells = [well for well in WELLS if well in set(simulation["well"])]
    nrows = int(np.ceil(len(available_wells) / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(15, 4 * nrows), sharex=True, squeeze=False)
    axes = axes.ravel()
    for axis, well in zip(axes, available_wells):
        color = COLORS[well]
        sim = simulation[simulation["well"] == well].sort_values("time_days")
        obs = observed[["time_days", OBS_COLUMNS[well]]].dropna()
        rmse = float(metrics.loc[metrics["well"] == well, "RMSE_m"].iloc[0])
        axis.plot(sim["time_days"], sim["sim_delta_head_m"], "--", color=color, linewidth=2.4, label="Simulation")
        axis.plot(obs["time_days"], obs[OBS_COLUMNS[well]], "-o", color=color, markerfacecolor="white", markersize=4.5, linewidth=1.5, label="Monitoring data")
        axis.axhline(0.0, color="#777777", linewidth=0.7)
        axis.set_title(f"{well} — RMSE {rmse:.3f} m", loc="left")
        axis.set_ylabel("Hydraulic-head change Δh (m)")
        axis.grid(True, alpha=0.3)
    for axis in axes[len(available_wells):]:
        axis.axis("off")
    for axis in axes[:len(available_wells)]:
        axis.set_xlabel("Time since 2019-01-13 (days)")
    fig.suptitle(
        f"{model}: monitoring-well hydraulic head — simulation vs monitoring",
        fontsize=20,
        y=0.995,
    )
    fig.legend(
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
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_model_one_axis(
    model: str,
    simulation: pd.DataFrame,
    observed: pd.DataFrame,
    metrics: pd.DataFrame,
    out: Path,
) -> None:
    """Overlay all available monitoring wells for one model on one axis."""
    available_wells = [well for well in WELLS if well in set(simulation["well"])]
    fig, axis = plt.subplots(figsize=(16, 9))
    for well in available_wells:
        color = COLORS[well]
        obs = observed[["time_days", OBS_COLUMNS[well]]].dropna()
        sim = simulation[simulation["well"] == well].sort_values("time_days")
        axis.plot(
            obs["time_days"],
            obs[OBS_COLUMNS[well]],
            "-o",
            color=color,
            markerfacecolor="white",
            markersize=4,
            linewidth=1.4,
        )
        axis.plot(
            sim["time_days"],
            sim["sim_delta_head_m"],
            "--",
            color=color,
            linewidth=2.1,
        )
    max_abs_head = float(np.nanmax(np.abs(simulation["sim_delta_head_m"])))
    scale_note = ""
    if max_abs_head > 1_000.0:
        axis.set_yscale("symlog", linthresh=1.0)
        scale_note = "; symmetric-log head scale"
    axis.axhline(0.0, color="#555555", linewidth=0.8)
    axis.set_title(f"{model}: all monitoring wells in one coordinate system{scale_note}")
    axis.set_xlabel("Time since 2019-01-13 (days)")
    axis.set_ylabel("Hydraulic-head change Δh (m)")
    axis.grid(True, which="both", alpha=0.25)
    well_handles = []
    for well in available_wells:
        rmse = float(metrics.loc[metrics["well"] == well, "RMSE_m"].iloc[0])
        well_handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS[well],
                linewidth=2,
                label=f"{well} (RMSE {rmse:.2f} m)",
            )
        )
    style_handles = [
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
    ]
    first_legend = axis.legend(
        handles=well_handles,
        title="Well color and RMSE",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )
    axis.add_artist(first_legend)
    axis.legend(
        handles=style_handles,
        title="Line style",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_all_models(simulations: pd.DataFrame, observed: pd.DataFrame, metrics: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(15, 16), sharex=True)
    axes = axes.ravel()
    for axis, well in zip(axes, WELLS):
        color = COLORS[well]
        obs = observed[["time_days", OBS_COLUMNS[well]]].dropna()
        axis.plot(obs["time_days"], obs[OBS_COLUMNS[well]], "-o", color=color, markerfacecolor="white", markersize=4.5, linewidth=1.6, label="Monitoring data")
        for model in MODEL_ORDER:
            sim = simulations[(simulations["model"] == model) & (simulations["well"] == well)].sort_values("time_days")
            if sim.empty:
                continue
            rmse = float(metrics[(metrics["model"] == model) & (metrics["well"] == well)]["RMSE_m"].iloc[0])
            axis.plot(sim["time_days"], sim["sim_delta_head_m"], color=color, linestyle=MODEL_STYLES[model], linewidth=2.1, alpha=MODEL_ALPHA[model], label=f"{model} simulation (RMSE {rmse:.2f} m)")
        axis.axhline(0.0, color="#777777", linewidth=0.7)
        axis.set_title(well, loc="left")
        axis.set_ylabel("Hydraulic-head change Δh (m)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8, loc="best")
    axes[-1].axis("off")
    for axis in axes[:-1]:
        axis.set_xlabel("Time since 2019-01-13 (days)")
    fig.suptitle("Four-model monitoring-well hydraulic-head comparison", fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_each_well_all_models(
    simulations: pd.DataFrame,
    observed: pd.DataFrame,
    metrics: pd.DataFrame,
    output: Path,
) -> None:
    """Write one shared-coordinate, cross-model figure for every monitoring well."""
    output.mkdir(exist_ok=True)
    for well in WELLS:
        fig, axis = plt.subplots(figsize=(10, 6))
        color = COLORS[well]
        obs = observed[["time_days", OBS_COLUMNS[well]]].dropna()
        axis.plot(
            obs["time_days"],
            obs[OBS_COLUMNS[well]],
            "-o",
            color=color,
            markerfacecolor="white",
            markersize=5,
            linewidth=1.7,
            label="Monitoring data",
        )
        for model in MODEL_ORDER:
            sim = simulations[
                (simulations["model"] == model) & (simulations["well"] == well)
            ].sort_values("time_days")
            if sim.empty:
                continue
            rmse = float(
                metrics[(metrics["model"] == model) & (metrics["well"] == well)][
                    "RMSE_m"
                ].iloc[0]
            )
            axis.plot(
                sim["time_days"],
                sim["sim_delta_head_m"],
                color=color,
                linestyle=MODEL_STYLES[model],
                linewidth=2.2,
                alpha=MODEL_ALPHA[model],
                label=f"{model} simulation (RMSE {rmse:.2f} m)",
            )
        axis.axhline(0.0, color="#777777", linewidth=0.7)
        axis.set_title(f"{well}: hydraulic-head comparison")
        axis.set_xlabel("Time since 2019-01-13 (days)")
        axis.set_ylabel("Hydraulic-head change Δh (m)")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output / f"{well}_all_models_head_comparison.png", dpi=220)
        plt.close(fig)


def plot_all_wells_one_axis(
    simulations: pd.DataFrame,
    observed: pd.DataFrame,
    out: Path,
) -> None:
    """Overlay every well and model in one coordinate system."""
    fig, axis = plt.subplots(figsize=(16, 9))
    for well in WELLS:
        color = COLORS[well]
        obs = observed[["time_days", OBS_COLUMNS[well]]].dropna()
        axis.plot(
            obs["time_days"],
            obs[OBS_COLUMNS[well]],
            "-o",
            color=color,
            markerfacecolor="white",
            markersize=3.5,
            linewidth=1.2,
            alpha=0.9,
        )
        for model in MODEL_ORDER:
            sim = simulations[
                (simulations["model"] == model) & (simulations["well"] == well)
            ].sort_values("time_days")
            if sim.empty:
                continue
            axis.plot(
                sim["time_days"],
                sim["sim_delta_head_m"],
                color=color,
                linestyle=MODEL_STYLES[model],
                linewidth=1.6,
                alpha=MODEL_ALPHA[model],
            )
    axis.axhline(0.0, color="#555555", linewidth=0.8)
    axis.set_yscale("symlog", linthresh=1.0)
    axis.set_title("All monitoring wells and models in one coordinate system")
    axis.set_xlabel("Time since 2019-01-13 (days)")
    axis.set_ylabel("Hydraulic-head change Δh (m; symmetric-log scale)")
    axis.grid(True, which="both", alpha=0.25)
    well_handles = [
        Line2D([0], [0], color=COLORS[well], linewidth=2, label=well) for well in WELLS
    ]
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", marker="o", markerfacecolor="white", label="Monitoring data"),
        *[
            Line2D([0], [0], color="black", linestyle=MODEL_STYLES[model], linewidth=2, label=f"{model} simulation")
            for model in MODEL_ORDER
        ],
    ]
    first_legend = axis.legend(handles=well_handles, title="Well color", loc="upper left", ncol=2)
    axis.add_artist(first_legend)
    axis.legend(handles=style_handles, title="Data/model style", loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
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
    all_mapping = []
    for model, settings in selected.items():
        simulation, mapping = extract_model(model, settings)
        comparison, metrics = compare(simulation, observed)
        output = HERE / str(settings.get("folder", model)) / "head_comparison"
        output.mkdir(exist_ok=True)
        simulation.to_csv(output / "simulated_monitoring_head_change.csv", index=False)
        comparison.to_csv(output / "simulation_vs_monitoring_head_change.csv", index=False)
        metrics.to_csv(output / "head_error_metrics.csv", index=False)
        mapping.to_csv(output / "monitoring_node_mapping.csv", index=False)
        plot_model(model, simulation, observed, metrics, output / "monitoring_well_head_comparison.png")
        plot_model_one_axis(
            model,
            simulation,
            observed,
            metrics,
            output / "all_monitoring_wells_one_coordinate.png",
        )
        all_simulation.append(simulation)
        all_comparison.append(comparison)
        all_metrics.append(metrics)
        all_mapping.append(mapping)
        print(f"Created hydraulic-head comparison for {model}")
    if args.model == "all":
        simulation = pd.concat(all_simulation, ignore_index=True)
        comparison = pd.concat(all_comparison, ignore_index=True)
        metrics = pd.concat(all_metrics, ignore_index=True)
        mapping = pd.concat(all_mapping, ignore_index=True)
        simulation.to_csv(HERE / "all_models_simulated_monitoring_head_change.csv", index=False)
        comparison.to_csv(HERE / "all_models_simulation_vs_monitoring_head_change.csv", index=False)
        metrics.to_csv(HERE / "all_models_head_error_metrics.csv", index=False)
        mapping.to_csv(HERE / "all_models_monitoring_node_mapping.csv", index=False)
        plot_all_models(simulation, observed, metrics, HERE / "all_models_monitoring_well_head_comparison.png")
        plot_each_well_all_models(
            simulation,
            observed,
            metrics,
            HERE / "head_comparison_by_well",
        )
        plot_all_wells_one_axis(
            simulation,
            observed,
            HERE / "all_wells_all_models_head_comparison_one_axis.png",
        )


if __name__ == "__main__":
    main()
