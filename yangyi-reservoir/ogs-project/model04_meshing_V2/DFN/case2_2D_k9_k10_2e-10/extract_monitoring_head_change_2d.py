#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract simulated hydraulic-head changes at monitoring wells from OGS VTU/PVD results.

Recommended use:
    python extract_monitoring_head_change_2d.py \
        --results Yangyi_2D_ZK403inj433_ZK208prod433_balanced.pvd \
        --monitoring monitoring_nodes_2d_fractures_filtered.csv \
        --outdir head_change_2d_balanced

Alternative, if you do not have a PVD file:
    python extract_monitoring_head_change_2d.py \
        --results "Yangyi_2D_ZK403inj433_ZK208prod433_balanced_ts_*.vtu" \
        --monitoring monitoring_nodes_2d_fractures_filtered.csv \
        --outdir head_change_2d_balanced

Outputs:
    simulated_head_timeseries.csv
    simulated_head_change_timeseries.csv
    simulated_head_change_long.csv
    simulated_head_change_*.png
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyvista as pv
except ImportError as exc:
    raise SystemExit(
        "ERROR: pyvista is required. Install it with:\n"
        "    pip install pyvista\n"
    ) from exc

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract simulated hydraulic head and head change at monitoring wells."
    )
    parser.add_argument(
        "--results",
        required=True,
        help="OGS .pvd file, one .vtu file, or a glob pattern such as 'result_ts_*.vtu'.",
    )
    parser.add_argument(
        "--monitoring",
        required=True,
        help="CSV file containing monitoring wells and selected node IDs/coordinates.",
    )
    parser.add_argument(
        "--outdir",
        default="head_change_output",
        help="Output directory.",
    )
    parser.add_argument(
        "--pressure-field",
        default="pressure",
        help="Name of pressure field in OGS VTU output. Default: pressure.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=1000.0,
        help="Fluid density in kg/m3. Default: 1000.",
    )
    parser.add_argument(
        "--g",
        type=float,
        default=9.81,
        help="Gravity acceleration in m/s2. Default: 9.81.",
    )
    parser.add_argument(
        "--prefer-node-id",
        action="store_true",
        help=(
            "Use selected_vtu_node_id directly. "
            "By default, the script uses coordinates and nearest-node search, "
            "which is safer after NodeReordering."
        ),
    )
    parser.add_argument(
        "--max-nearest-distance",
        type=float,
        default=5.0,
        help=(
            "Warning threshold for nearest-node distance in meters. "
            "Default: 5 m. This does not stop the script."
        ),
    )
    return parser.parse_args()


def parse_time_from_filename(path: str | Path) -> float:
    """
    Fallback time parser for VTU filenames.
    Tries common OGS patterns such as:
        *_ts_10_t_86400.000000.vtu
        *_t_86400.vtu
        *_time_86400.vtu
    Returns NaN if no time is found.
    """
    name = Path(path).name

    patterns = [
        r"(?:^|[_-])t[_-]?([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)(?:[_\.]|$)",
        r"(?:^|[_-])time[_-]?([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)(?:[_\.]|$)",
        r"(?:^|[_-])ts[_-]?[0-9]+[_-]t[_-]?([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)(?:[_\.]|$)",
    ]

    for pat in patterns:
        m = re.search(pat, name)
        if m:
            return float(m.group(1))

    return math.nan


def read_pvd(pvd_path: str | Path) -> list[tuple[float, Path]]:
    """
    Read a PVD file and return [(time, vtu_path), ...].
    Time is assumed to be the OGS timestep value, usually in seconds.
    """
    pvd_path = Path(pvd_path)
    tree = ET.parse(pvd_path)
    root = tree.getroot()

    records: list[tuple[float, Path]] = []
    for dataset in root.iter("DataSet"):
        timestep = float(dataset.attrib.get("timestep", "nan"))
        file_attr = dataset.attrib.get("file")
        if not file_attr:
            continue
        vtu_path = (pvd_path.parent / file_attr).resolve()
        records.append((timestep, vtu_path))

    records.sort(key=lambda x: x[0])
    return records


def collect_result_files(results_arg: str) -> list[tuple[float, Path]]:
    path = Path(results_arg)

    if path.suffix.lower() == ".pvd" and path.exists():
        records = read_pvd(path)
    else:
        matches = sorted(glob.glob(results_arg))
        if not matches and path.exists() and path.suffix.lower() == ".vtu":
            matches = [str(path)]
        if not matches:
            raise FileNotFoundError(f"No result files found from: {results_arg}")

        records = [(parse_time_from_filename(p), Path(p).resolve()) for p in matches]

        # If times cannot be parsed, use file order as timestep index.
        if all(math.isnan(t) for t, _ in records):
            records = [(float(i), p) for i, (_, p) in enumerate(records)]
        else:
            records = sorted(records, key=lambda x: (math.inf if math.isnan(x[0]) else x[0]))

    if not records:
        raise RuntimeError("No VTU records found.")

    missing = [str(p) for _, p in records if not p.exists()]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(f"Some VTU files referenced by results are missing:\n{preview}")

    return records


def find_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise KeyError(f"Could not find any of these columns: {candidates}\nAvailable: {list(df.columns)}")
    return None


def load_monitoring_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    well_col = find_column(df, ["Well", "well", "Name", "name"])
    x_col = find_column(df, ["selected_x", "x", "X", "POINT_X"])
    y_col = find_column(df, ["selected_y", "y", "Y", "POINT_Y"])
    z_col = find_column(df, ["selected_z", "z", "Z"])

    node_col = find_column(
        df,
        ["selected_vtu_node_id", "node_id", "NodeID", "selected_node_id", "vtkOriginalPointId"],
        required=False,
    )

    out = pd.DataFrame()
    out["well"] = df[well_col].astype(str)
    out["x"] = pd.to_numeric(df[x_col], errors="raise")
    out["y"] = pd.to_numeric(df[y_col], errors="raise")
    out["z"] = pd.to_numeric(df[z_col], errors="raise")
    if node_col is not None:
        out["node_id"] = pd.to_numeric(df[node_col], errors="coerce").astype("Int64")
    else:
        out["node_id"] = pd.Series([pd.NA] * len(out), dtype="Int64")

    return out


def get_pressure_array(mesh: pv.DataSet, pressure_field: str) -> np.ndarray:
    if pressure_field in mesh.point_data:
        arr = np.asarray(mesh.point_data[pressure_field])
        location = "point"
    elif pressure_field in mesh.cell_data:
        raise RuntimeError(
            f"Pressure field '{pressure_field}' is in cell_data, not point_data. "
            "This script expects nodal pressure for monitoring-node extraction. "
            "Please output pressure as point data or convert cell data to point data."
        )
    else:
        available = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
        raise KeyError(
            f"Pressure field '{pressure_field}' not found. Available fields: {available}"
        )

    arr = np.asarray(arr).reshape(-1)
    if len(arr) != mesh.n_points:
        raise RuntimeError(
            f"Pressure array length {len(arr)} does not match number of points {mesh.n_points}."
        )
    return arr


def select_nodes(
    mesh: pv.DataSet,
    monitors: pd.DataFrame,
    prefer_node_id: bool,
    max_nearest_distance: float,
) -> pd.DataFrame:
    points = np.asarray(mesh.points)
    selected = []

    for _, row in monitors.iterrows():
        coord = np.array([row["x"], row["y"], row["z"]], dtype=float)

        used_method = "nearest_coordinate"
        node_id = None
        nearest_distance = None

        if prefer_node_id and pd.notna(row["node_id"]):
            cand = int(row["node_id"])
            if 0 <= cand < mesh.n_points:
                node_id = cand
                nearest_distance = float(np.linalg.norm(points[node_id] - coord))
                used_method = "node_id"
            else:
                print(
                    f"WARNING: {row['well']} node_id={cand} is outside mesh point range. "
                    "Falling back to nearest coordinate."
                )

        if node_id is None:
            # pyvista find_closest_point is robust and avoids building scipy dependency.
            node_id = int(mesh.find_closest_point(coord))
            nearest_distance = float(np.linalg.norm(points[node_id] - coord))

        if nearest_distance > max_nearest_distance:
            print(
                f"WARNING: {row['well']} nearest node is {nearest_distance:.3f} m away "
                f"from monitoring coordinate. Please check mesh consistency."
            )

        px, py, pz = points[node_id]
        selected.append(
            {
                "well": row["well"],
                "node_id": node_id,
                "input_x": row["x"],
                "input_y": row["y"],
                "input_z": row["z"],
                "mesh_x": px,
                "mesh_y": py,
                "mesh_z": pz,
                "nearest_distance_m": nearest_distance,
                "selection_method": used_method,
            }
        )

    return pd.DataFrame(selected)


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = collect_result_files(args.results)
    monitors = load_monitoring_table(args.monitoring)

    first_mesh = pv.read(records[0][1])
    selected_nodes = select_nodes(
        first_mesh,
        monitors,
        prefer_node_id=args.prefer_node_id,
        max_nearest_distance=args.max_nearest_distance,
    )
    selected_nodes.to_csv(outdir / "selected_monitoring_nodes_used.csv", index=False)

    print("Selected monitoring nodes:")
    print(selected_nodes[["well", "node_id", "mesh_x", "mesh_y", "mesh_z", "nearest_distance_m"]])

    rows = []

    for time_value, vtu_path in records:
        mesh = pv.read(vtu_path)
        pressure = get_pressure_array(mesh, args.pressure_field)
        points = np.asarray(mesh.points)

        for _, mon in selected_nodes.iterrows():
            nid = int(mon["node_id"])
            p = float(pressure[nid])
            z = float(points[nid, 2])
            head = p / (args.rho * args.g) + z

            rows.append(
                {
                    "time": time_value,
                    "time_days": time_value / 86400.0,
                    "file": vtu_path.name,
                    "well": mon["well"],
                    "node_id": nid,
                    "x": float(points[nid, 0]),
                    "y": float(points[nid, 1]),
                    "z": z,
                    "pressure_Pa": p,
                    "head_m": head,
                }
            )

    long_df = pd.DataFrame(rows)
    long_df = long_df.sort_values(["well", "time"]).reset_index(drop=True)

    # Relative head change for each well
    long_df["head_initial_m"] = long_df.groupby("well")["head_m"].transform("first")
    long_df["delta_head_m"] = long_df["head_m"] - long_df["head_initial_m"]
    long_df["delta_pressure_Pa"] = long_df.groupby("well")["pressure_Pa"].transform(lambda s: s - s.iloc[0])
    long_df["delta_pressure_MPa"] = long_df["delta_pressure_Pa"] / 1.0e6

    long_path = outdir / "simulated_head_change_long.csv"
    long_df.to_csv(long_path, index=False)

    # Wide tables
    head_wide = long_df.pivot_table(index=["time", "time_days"], columns="well", values="head_m").reset_index()
    dh_wide = long_df.pivot_table(index=["time", "time_days"], columns="well", values="delta_head_m").reset_index()
    p_wide = long_df.pivot_table(index=["time", "time_days"], columns="well", values="pressure_Pa").reset_index()
    dp_wide = long_df.pivot_table(index=["time", "time_days"], columns="well", values="delta_pressure_MPa").reset_index()

    head_wide.to_csv(outdir / "simulated_head_timeseries.csv", index=False)
    dh_wide.to_csv(outdir / "simulated_head_change_timeseries.csv", index=False)
    p_wide.to_csv(outdir / "simulated_pressure_timeseries.csv", index=False)
    dp_wide.to_csv(outdir / "simulated_pressure_change_MPa_timeseries.csv", index=False)

    # Summary
    summary = (
        long_df.groupby("well")
        .agg(
            node_id=("node_id", "first"),
            x=("x", "first"),
            y=("y", "first"),
            z=("z", "first"),
            head_initial_m=("head_initial_m", "first"),
            delta_head_min_m=("delta_head_m", "min"),
            delta_head_max_m=("delta_head_m", "max"),
            delta_head_final_m=("delta_head_m", "last"),
            pressure_initial_Pa=("pressure_Pa", "first"),
            delta_pressure_final_MPa=("delta_pressure_MPa", "last"),
        )
        .reset_index()
    )
    summary.to_csv(outdir / "simulated_head_change_summary.csv", index=False)

    # Plot one figure with all wells
    plt.figure(figsize=(9, 5.5))
    for well, sub in long_df.groupby("well"):
        plt.plot(sub["time_days"], sub["delta_head_m"], label=well)
    plt.xlabel("Time (days)")
    plt.ylabel("Hydraulic head change Δh (m)")
    plt.title("Simulated monitoring-well head change")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "simulated_head_change_all_wells.png", dpi=300)
    plt.close()

    # Plot separate figures
    for well, sub in long_df.groupby("well"):
        plt.figure(figsize=(7, 4.5))
        plt.plot(sub["time_days"], sub["delta_head_m"])
        plt.xlabel("Time (days)")
        plt.ylabel("Hydraulic head change Δh (m)")
        plt.title(f"{well} simulated head change")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_well = re.sub(r"[^A-Za-z0-9_-]+", "_", str(well))
        plt.savefig(outdir / f"simulated_head_change_{safe_well}.png", dpi=300)
        plt.close()

    print("\nDone.")
    print(f"Output directory: {outdir.resolve()}")
    print(f"Main long table: {long_path}")
    print("Key output: simulated_head_change_timeseries.csv")


if __name__ == "__main__":
    main()
