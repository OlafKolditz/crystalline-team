#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare simulated and observed monitoring-well hydraulic-head changes for the
Yangyi actual injection/production case after 2019-01-13.

This version keeps the original monitoring-point CSV unchanged:

    monitoring/recommended_monitoring_points_all7_ZK501_bedrock.csv

Only the simulation/observation inputs are changed to:

    Yangyi_actual_ops_after20190113.pvd
    yangyi_after_20190113_for_ogs.xlsx

Important time convention
-------------------------
The OGS simulation time zero is 2019-01-13 00:00:00.  The observed monitoring
head changes in the Excel file are already re-zeroed to the first monitoring
record after filtering, generally 2019-01-14.  Therefore, by default this script
also re-zeroes simulated head at the first observation time in the selected Excel
sheet, instead of at simulation day 0.

Default observed sheet:

    Monitor_compare_after20190113

Examples
--------
Single-line command, recommended to avoid shell backslash/space issues:

    python compare_simulated_observed_head_change_after20190113.py --pvd Yangyi_actual_ops_after20190113.pvd --monitor_csv monitoring/recommended_monitoring_points_all7_ZK501_bedrock.csv --obs_xlsx yangyi_after_20190113_for_ogs.xlsx

If your VTU output stores pressure as CellData, this script will automatically
extract pressure from the closest cell to each monitoring point.
"""
from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import pyvista as pv
except Exception:  # pragma: no cover
    pv = None

try:
    import meshio
except Exception:  # pragma: no cover
    meshio = None


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def safe(s: object) -> str:
    return re.sub(r"[^A-Za-z0-9_\-.]+", "_", str(s)).strip("_") or "plot"


def parse_pvd(pvd_path: Path) -> List[Tuple[float, Path]]:
    if not pvd_path.exists():
        raise FileNotFoundError(f"PVD file not found: {pvd_path}")
    root = ET.parse(pvd_path).getroot()
    out: List[Tuple[float, Path]] = []
    for ds in root.findall(".//DataSet"):
        if "file" not in ds.attrib:
            continue
        timestep = float(ds.attrib.get("timestep", "nan"))
        vtu = (pvd_path.parent / ds.attrib["file"]).resolve()
        out.append((timestep, vtu))
    out = sorted(out, key=lambda x: x[0])
    if not out:
        raise RuntimeError(f"No VTU datasets found in {pvd_path}")
    return out


def time_from_vtu_name(path: Path) -> float:
    """Best-effort parser for --pattern mode when no PVD is available."""
    name = path.name
    patterns = [
        r"_ts_([0-9]+(?:\.[0-9]+)?)",
        r"_t_([0-9]+(?:\.[0-9]+)?)",
        r"time[_-]?([0-9]+(?:\.[0-9]+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            return float(m.group(1))
    nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)", name)
    if nums:
        return float(nums[-1])
    raise RuntimeError(f"Cannot parse time from VTU filename: {name}. Use --pvd instead.")


def collect_results(pvd: Optional[str], pattern: Optional[str]) -> List[Tuple[float, Path]]:
    if pvd:
        return parse_pvd(Path(pvd))
    if not pattern:
        raise RuntimeError("Please provide --pvd or --pattern")
    files = sorted(Path(".").glob(pattern))
    if not files:
        files = sorted(Path(".").glob(str(pattern)))
    if not files:
        raise RuntimeError(f"No VTU files found with pattern: {pattern}")
    return sorted([(time_from_vtu_name(f), f.resolve()) for f in files], key=lambda x: x[0])


# -----------------------------------------------------------------------------
# Reading VTU outputs
# -----------------------------------------------------------------------------

def read_vtu(path: Path):
    if pv is not None:
        mesh = pv.read(str(path))
        point_data = {k: np.asarray(v) for k, v in mesh.point_data.items()}
        cell_data = {k: np.asarray(v) for k, v in mesh.cell_data.items()}
        return mesh, np.asarray(mesh.points), point_data, cell_data

    if meshio is not None:
        mesh = meshio.read(str(path))
        point_data = {k: np.asarray(v) for k, v in mesh.point_data.items()}
        cell_data: Dict[str, np.ndarray] = {}
        for k, blocks in mesh.cell_data.items():
            try:
                cell_data[k] = np.concatenate([np.asarray(b).reshape(-1) for b in blocks])
            except Exception:
                pass
        return mesh, np.asarray(mesh.points), point_data, cell_data

    raise RuntimeError("Please install pyvista or meshio: pip install pyvista meshio")


def choose_array(data: Dict[str, np.ndarray], preferred: Optional[str], role: str) -> str:
    if preferred:
        if preferred not in data:
            raise RuntimeError(f"{preferred!r} not found in {role}; available={list(data)}")
        return preferred
    for n in ["pressure", "p", "liquid_pressure", "pressure_interpolated"]:
        if n in data:
            return n
    for n in data:
        if "pressure" in n.lower():
            return n
    raise RuntimeError(f"Cannot find pressure array in {role}; available={list(data)}")


def cell_centers_meshio(mesh) -> np.ndarray:
    centers = []
    pts = np.asarray(mesh.points)
    for block in mesh.cells:
        conn = np.asarray(block.data)
        if conn.size == 0:
            continue
        centers.append(pts[conn].mean(axis=1))
    if not centers:
        raise RuntimeError("Cannot compute cell centers from meshio mesh")
    return np.vstack(centers)


def find_closest_cell_id(mesh, centers: Optional[np.ndarray], xyz: np.ndarray) -> int:
    if pv is not None and hasattr(mesh, "find_closest_cell"):
        return int(mesh.find_closest_cell(xyz))
    if centers is None:
        centers = cell_centers_meshio(mesh)
    d2 = np.sum((centers - xyz.reshape(1, 3)) ** 2, axis=1)
    return int(np.argmin(d2))


# -----------------------------------------------------------------------------
# Monitoring and observations
# -----------------------------------------------------------------------------

def make_point_label(row: pd.Series) -> str:
    parts = [str(row.get("well", ""))]
    rt = str(row.get("recommended_type", ""))
    if rt and rt.lower() != "nan":
        parts.append(rt)
    mid = row.get("MaterialID", "")
    if str(mid) not in ["", "nan"]:
        try:
            parts.append(f"MID{int(float(mid))}")
        except Exception:
            parts.append(f"MID{mid}")
    dep = row.get("depth_m", "")
    if str(dep) not in ["", "nan"]:
        try:
            parts.append(f"{float(dep):.1f} m")
        except Exception:
            pass
    return " | ".join([p for p in parts if p])


def read_monitor_csv(monitor_csv: Path) -> pd.DataFrame:
    if not monitor_csv.exists():
        raise FileNotFoundError(f"monitor_csv not found: {monitor_csv}")
    mon = pd.read_csv(monitor_csv)

    # Keep compatibility with your original recommended monitoring CSV.
    lower = {c.lower(): c for c in mon.columns}
    if "well" not in mon.columns and "well" in lower:
        mon = mon.rename(columns={lower["well"]: "well"})
    if "bulk_node_id" not in mon.columns:
        for cand in ["bulk_node_ids", "bulk_node", "node_id", "node", "bulk_id"]:
            if cand in lower:
                mon = mon.rename(columns={lower[cand]: "bulk_node_id"})
                break

    if "well" not in mon.columns or "bulk_node_id" not in mon.columns:
        raise RuntimeError(
            "monitor_csv must contain well and bulk_node_id columns. "
            f"Current columns: {mon.columns.tolist()}"
        )

    mon = mon.copy()
    mon["well"] = mon["well"].astype(str).str.strip()
    mon["bulk_node_id"] = mon["bulk_node_id"].astype(int)
    mon["point_label"] = mon.apply(make_point_label, axis=1)
    return mon


def read_observed_from_xlsx(obs_xlsx: Path, sheet: str) -> pd.DataFrame:
    if not obs_xlsx.exists():
        raise FileNotFoundError(f"obs_xlsx not found: {obs_xlsx}")
    obs = pd.read_excel(obs_xlsx, sheet_name=sheet)
    if "Date" not in obs.columns:
        raise RuntimeError(f"Observed sheet {sheet!r} must contain Date column")

    if "time_days_from_20190113" in obs.columns:
        time_col = "time_days_from_20190113"
    elif "time_days" in obs.columns:
        time_col = "time_days"
    else:
        raise RuntimeError(
            f"Observed sheet {sheet!r} must contain time_days_from_20190113 or time_days. "
            f"Columns: {obs.columns.tolist()}"
        )

    out = pd.DataFrame({
        "Date": pd.to_datetime(obs["Date"]),
        "time_days": pd.to_numeric(obs[time_col], errors="coerce"),
    })

    # Convert columns like SC211_obs_delta_h_m_from_filtered_baseline to SC211_delta_h_m.
    for c in obs.columns:
        m = re.match(r"^(.+?)_obs_delta_h_m_from_filtered_baseline$", str(c))
        if m:
            well = m.group(1)
            out[f"{well}_delta_h_m"] = pd.to_numeric(obs[c], errors="coerce")
            continue
        m = re.match(r"^(.+?)_delta_h_m_from_filtered_baseline$", str(c))
        if m:
            well = m.group(1)
            out[f"{well}_delta_h_m"] = pd.to_numeric(obs[c], errors="coerce")

    delta_cols = [c for c in out.columns if c.endswith("_delta_h_m")]
    if not delta_cols:
        raise RuntimeError(
            f"No observed delta-head columns found in {sheet!r}. Expected columns like "
            "SC211_obs_delta_h_m_from_filtered_baseline."
        )
    out = out.dropna(subset=["time_days"]).sort_values("time_days").reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Extraction and comparison
# -----------------------------------------------------------------------------

def extract_sim(
    results: List[Tuple[float, Path]],
    monitor_csv: Path,
    rho: float,
    g: float,
    time_scale: float,
    pressure_name: Optional[str],
) -> pd.DataFrame:
    mon = read_monitor_csv(monitor_csv)
    rows = []
    pname = None
    pressure_location = None
    centers_cache = None

    for traw, vtu in results:
        if not Path(vtu).exists():
            raise FileNotFoundError(f"VTU file not found: {vtu}")
        mesh, pts, point_data, cell_data = read_vtu(Path(vtu))

        if pname is None:
            # Prefer point data, but fall back to cell data if pressure is written as CellData.
            if pressure_name and pressure_name in point_data:
                pname = pressure_name
                pressure_location = "point"
            elif pressure_name and pressure_name in cell_data:
                pname = pressure_name
                pressure_location = "cell"
            else:
                try:
                    pname = choose_array(point_data, pressure_name, "PointData")
                    pressure_location = "point"
                except RuntimeError:
                    pname = choose_array(cell_data, pressure_name, "CellData")
                    pressure_location = "cell"
            print(f"Using pressure array: {pname} ({pressure_location} data)")

        if pressure_location == "point":
            pres = np.asarray(point_data[pname]).reshape(-1)
            for _, r in mon.iterrows():
                nid = int(r["bulk_node_id"])
                if nid >= len(pts):
                    raise RuntimeError(f"bulk_node_id {nid} exceeds node count {len(pts)} in {Path(vtu).name}")
                z = float(pts[nid][2])
                p = float(pres[nid])
                h = p / (rho * g) + z
                rows.append({
                    "time_days": traw / time_scale,
                    "well": r["well"],
                    "point_label": r["point_label"],
                    "bulk_node_id": nid,
                    "extract_id": nid,
                    "extract_location": "point",
                    "pressure_Pa": p,
                    "head_m": h,
                    "source_vtu": Path(vtu).name,
                })
        else:
            pres = np.asarray(cell_data[pname]).reshape(-1)
            if pv is None and centers_cache is None:
                centers_cache = cell_centers_meshio(mesh)
            for _, r in mon.iterrows():
                nid = int(r["bulk_node_id"])
                if nid >= len(pts):
                    raise RuntimeError(f"bulk_node_id {nid} exceeds node count {len(pts)} in {Path(vtu).name}")
                xyz = np.asarray(pts[nid], dtype=float)
                cid = find_closest_cell_id(mesh, centers_cache, xyz)
                if cid >= len(pres):
                    raise RuntimeError(f"closest cell id {cid} exceeds cell pressure length {len(pres)} in {Path(vtu).name}")
                z = float(xyz[2])
                p = float(pres[cid])
                h = p / (rho * g) + z
                rows.append({
                    "time_days": traw / time_scale,
                    "well": r["well"],
                    "point_label": r["point_label"],
                    "bulk_node_id": nid,
                    "extract_id": cid,
                    "extract_location": "closest_cell",
                    "pressure_Pa": p,
                    "head_m": h,
                    "source_vtu": Path(vtu).name,
                })

    if not rows:
        raise RuntimeError("No simulated monitoring records extracted")
    return pd.DataFrame(rows)


def add_zeroed_delta(sim: pd.DataFrame, zero_time_days: float) -> pd.DataFrame:
    out = []
    for label, gdf in sim.groupby("point_label", sort=False):
        gdf = gdf.sort_values("time_days").copy()
        t = gdf["time_days"].to_numpy(float)
        h = gdf["head_m"].to_numpy(float)
        if zero_time_days < t.min() or zero_time_days > t.max():
            raise RuntimeError(
                f"zero_time_days={zero_time_days} outside simulation time range {t.min()}..{t.max()} "
                f"for {label}"
            )
        h0 = float(np.interp(zero_time_days, t, h))
        gdf["sim_head_zero_m"] = h0
        gdf["sim_delta_h_m"] = h - h0
        gdf["sim_zero_time_days"] = zero_time_days
        out.append(gdf)
    return pd.concat(out, ignore_index=True)


def interpolate_to_obs(sim_zero: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    obs_times = obs["time_days"].to_numpy(float)
    for label, gdf in sim_zero.groupby("point_label", sort=False):
        well = str(gdf["well"].iloc[0])
        gdf = gdf.sort_values("time_days")
        obs_col = f"{well}_delta_h_m"
        if obs_col not in obs.columns:
            print(f"WARNING: observed column not found for {well}: {obs_col}; skipped.")
            continue
        sim_interp = np.interp(
            obs_times,
            gdf["time_days"].to_numpy(float),
            gdf["sim_delta_h_m"].to_numpy(float),
        )
        for i, row in obs.iterrows():
            obs_val = row.get(obs_col, np.nan)
            rows.append({
                "Date": row.get("Date"),
                "time_days_from_20190113": float(row["time_days"]),
                "well": well,
                "point_label": label,
                "bulk_node_id": int(gdf["bulk_node_id"].iloc[0]),
                "extract_location": str(gdf["extract_location"].iloc[0]),
                "extract_id": int(gdf["extract_id"].iloc[0]),
                "sim_delta_h_m": float(sim_interp[i]),
                "obs_delta_h_m": obs_val,
                "residual_sim_minus_obs_m": float(sim_interp[i]) - obs_val if pd.notna(obs_val) else np.nan,
            })
    if not rows:
        raise RuntimeError("No comparison rows were generated. Check well names and observed columns.")
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plots and summary
# -----------------------------------------------------------------------------

def plot_well_comparison(comp: pd.DataFrame, out_dir: Path, zero_time_days: float):
    for well, wdf in comp.groupby("well", sort=False):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        obs = wdf.drop_duplicates("time_days_from_20190113").sort_values("time_days_from_20190113")
        ax.plot(
            obs["time_days_from_20190113"],
            obs["obs_delta_h_m"],
            marker="o",
            linewidth=2.0,
            label=f"{well} observed",
        )
        for label, gdf in wdf.groupby("point_label", sort=False):
            gdf = gdf.sort_values("time_days_from_20190113")
            ax.plot(
                gdf["time_days_from_20190113"],
                gdf["sim_delta_h_m"],
                marker="s",
                markersize=3,
                linewidth=1.3,
                linestyle="--",
                label=f"sim: {label}",
            )
        ax.axhline(0, linewidth=0.8)
        ax.axvline(zero_time_days, linewidth=0.8, linestyle=":")
        ax.set_xlabel("Time from 2019-01-13 (days)")
        ax.set_ylabel(f"Head change Δh (m), zeroed at day {zero_time_days:g}")
        ax.set_title(f"{well}: simulated vs observed head change after 2019-01-13")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"compare_delta_head_{safe(well)}.png", dpi=300)
        plt.close(fig)


def plot_all_mean(comp: pd.DataFrame, out_png: Path, zero_time_days: float):
    sim_mean = comp.groupby(["time_days_from_20190113", "well"], as_index=False)["sim_delta_h_m"].mean()
    obs_once = comp.drop_duplicates(["time_days_from_20190113", "well"])
    fig, ax = plt.subplots(figsize=(11, 6))
    for well, gdf in obs_once.groupby("well", sort=False):
        gdf = gdf.sort_values("time_days_from_20190113")
        ax.plot(gdf["time_days_from_20190113"], gdf["obs_delta_h_m"], marker="o", linewidth=1.8, label=f"{well} obs")
    for well, gdf in sim_mean.groupby("well", sort=False):
        gdf = gdf.sort_values("time_days_from_20190113")
        ax.plot(gdf["time_days_from_20190113"], gdf["sim_delta_h_m"], linewidth=1.4, linestyle="--", label=f"{well} sim")
    ax.axhline(0, linewidth=0.8)
    ax.axvline(zero_time_days, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Time from 2019-01-13 (days)")
    ax.set_ylabel("Head change Δh (m)")
    ax.set_title("Observed vs simulated monitoring-well head change after 2019-01-13")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def make_summary(comp: pd.DataFrame) -> pd.DataFrame:
    return comp.groupby("well", as_index=False).agg(
        n_obs=("obs_delta_h_m", "count"),
        mean_obs_delta_h_m=("obs_delta_h_m", "mean"),
        mean_sim_delta_h_m=("sim_delta_h_m", "mean"),
        mean_residual_m=("residual_sim_minus_obs_m", "mean"),
        mae_m=("residual_sim_minus_obs_m", lambda x: float(np.nanmean(np.abs(x)))),
        rmse_m=("residual_sim_minus_obs_m", lambda x: float(np.sqrt(np.nanmean(np.square(x))))),
        max_abs_error_m=("residual_sim_minus_obs_m", lambda x: float(np.nanmax(np.abs(x)))),
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pvd", default="Yangyi_actual_ops_after20190113.pvd", help="OGS PVD file")
    ap.add_argument("--pattern", default=None, help="VTU glob pattern used only if --pvd is empty")
    ap.add_argument("--monitor_csv", default="monitoring/recommended_monitoring_points_all7_ZK501_bedrock.csv")
    ap.add_argument("--obs_xlsx", default="yangyi_after_20190113_for_ogs.xlsx")
    ap.add_argument("--obs_sheet", default="Monitor_compare_after20190113")
    ap.add_argument("--out_dir", default="comparison_head_change_after20190113")
    ap.add_argument(
        "--zero_time_days",
        type=float,
        default=None,
        help="Simulation re-zero time in days from 2019-01-13. Default: first observation time in obs sheet.",
    )
    ap.add_argument("--rho", type=float, default=1000.0)
    ap.add_argument("--g", type=float, default=9.81)
    ap.add_argument("--time_scale", type=float, default=86400.0, help="Convert PVD timestep seconds to days")
    ap.add_argument("--pressure_name", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs = read_observed_from_xlsx(Path(args.obs_xlsx), args.obs_sheet)
    zero_time_days = float(args.zero_time_days) if args.zero_time_days is not None else float(obs["time_days"].min())
    print(f"Observed data: {args.obs_xlsx} / {args.obs_sheet}")
    print(f"Simulation zero time: 2019-01-13; Δh re-zeroed at day {zero_time_days:g}")

    pvd_arg = args.pvd if args.pvd and str(args.pvd).lower() not in ["none", ""] else None
    results = collect_results(pvd_arg, args.pattern)
    sim = extract_sim(results, Path(args.monitor_csv), args.rho, args.g, args.time_scale, args.pressure_name)
    sim0 = add_zeroed_delta(sim, zero_time_days)
    comp = interpolate_to_obs(sim0, obs)
    summary = make_summary(comp)

    sim.to_csv(out_dir / "simulated_monitor_heads_raw.csv", index=False, encoding="utf-8-sig")
    sim0.to_csv(out_dir / "simulated_head_change_all_outputs.csv", index=False, encoding="utf-8-sig")
    obs.to_csv(out_dir / "observed_head_change_from_excel.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(out_dir / "simulated_vs_observed_delta_head_at_observation_times.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "comparison_error_summary_by_well.csv", index=False, encoding="utf-8-sig")

    plot_well_comparison(comp, out_dir, zero_time_days)
    plot_all_mean(comp, out_dir / "compare_delta_head_all_wells.png", zero_time_days)

    print("Saved:", out_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
