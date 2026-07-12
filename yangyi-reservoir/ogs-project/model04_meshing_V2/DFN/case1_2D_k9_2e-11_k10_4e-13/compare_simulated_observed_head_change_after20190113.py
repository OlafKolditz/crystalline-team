#!/usr/bin/env python3
"""
Compare 2D simulated monitoring-well hydraulic-head changes with the filtered
Yangyi monitoring data after 2019-01-13.

The OGS simulation starts at 2019-01-13 00:00:00. The observed monitoring
changes in Monitor_compare_after20190113 are re-zeroed at the first monitoring
record after filtering, usually simulation day 1. This script therefore reads
absolute simulated heads and re-zeroes each simulated well at the first
observation time before comparing.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_WELLS = ["ZK204", "ZK206", "ZK207", "ZK401", "ZK402"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare 2D after-2019-01-13 simulated and observed head changes.")
    p.add_argument("--sim-heads", required=True, help="simulated_head_timeseries.csv from extraction script.")
    p.add_argument("--obs-xlsx", default="../../case6_match_SC211_ZK207/yangyi_after_20190113_for_ogs.xlsx")
    p.add_argument("--obs-sheet", default="Monitor_compare_after20190113")
    p.add_argument("--outdir", default="comparison_after20190113_2d", help="Output directory")
    p.add_argument("--wells", nargs="*", default=DEFAULT_WELLS)
    p.add_argument("--time-col", default="time_days")
    return p.parse_args()


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    den = np.sum((obs - np.mean(obs)) ** 2)
    if den == 0:
        return np.nan
    return float(1.0 - np.sum((sim - obs) ** 2) / den)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sim_heads = pd.read_csv(args.sim_heads)
    obs = pd.read_excel(args.obs_xlsx, sheet_name=args.obs_sheet)

    if args.time_col not in sim_heads.columns:
        raise KeyError(f"Simulation time column {args.time_col!r} not found: {list(sim_heads.columns)}")
    if "time_days_from_20190113" not in obs.columns:
        raise KeyError(f"Observed sheet must contain time_days_from_20190113: {list(obs.columns)}")

    rows = []
    metrics = []

    for well in args.wells:
        sim_col = well
        obs_col = f"{well}_obs_delta_h_m_from_filtered_baseline"
        if sim_col not in sim_heads.columns:
            print(f"WARNING: simulated column missing for {well}; skipped")
            continue
        if obs_col not in obs.columns:
            print(f"WARNING: observed column missing for {well}; skipped")
            continue

        sim_sub = sim_heads[[args.time_col, sim_col]].dropna().sort_values(args.time_col)
        obs_sub = obs[["Date", "time_days_from_20190113", obs_col]].dropna().sort_values("time_days_from_20190113")
        if sim_sub.empty or obs_sub.empty:
            print(f"WARNING: no data for {well}; skipped")
            continue

        t_sim = sim_sub[args.time_col].to_numpy(dtype=float)
        h_sim = sim_sub[sim_col].to_numpy(dtype=float)
        t_obs = obs_sub["time_days_from_20190113"].to_numpy(dtype=float)
        y_obs = obs_sub[obs_col].to_numpy(dtype=float)

        h_at_obs = np.interp(t_obs, t_sim, h_sim)
        h_at_obs[(t_obs < np.min(t_sim)) | (t_obs > np.max(t_sim))] = np.nan
        h0 = np.interp(float(t_obs[0]), t_sim, h_sim)
        y_sim = h_at_obs - h0

        comp = pd.DataFrame(
            {
                "well": well,
                "Date": obs_sub["Date"].values,
                "time_days_from_20190113": t_obs,
                "observed_delta_h_m": y_obs,
                "simulated_delta_h_m": y_sim,
            }
        )
        comp["residual_sim_minus_obs_m"] = comp["simulated_delta_h_m"] - comp["observed_delta_h_m"]
        rows.append(comp)

        valid = comp.dropna(subset=["observed_delta_h_m", "simulated_delta_h_m"])
        if not valid.empty:
            residual = valid["residual_sim_minus_obs_m"].to_numpy(dtype=float)
            obs_vals = valid["observed_delta_h_m"].to_numpy(dtype=float)
            sim_vals = valid["simulated_delta_h_m"].to_numpy(dtype=float)
            metrics.append(
                {
                    "well": well,
                    "n_points": len(valid),
                    "mean_obs_delta_h_m": float(np.mean(obs_vals)),
                    "mean_sim_delta_h_m": float(np.mean(sim_vals)),
                    "mean_residual_m": float(np.mean(residual)),
                    "mae_m": float(np.mean(np.abs(residual))),
                    "rmse_m": float(np.sqrt(np.mean(residual**2))),
                    "max_abs_error_m": float(np.max(np.abs(residual))),
                    "corr": float(np.corrcoef(obs_vals, sim_vals)[0, 1]) if len(valid) >= 2 else np.nan,
                    "nse": nse(obs_vals, sim_vals),
                    "final_obs_m": float(obs_vals[-1]),
                    "final_sim_m": float(sim_vals[-1]),
                }
            )

    if not rows:
        raise RuntimeError("No valid wells were compared")

    long = pd.concat(rows, ignore_index=True)
    long.to_csv(outdir / "comparison_sim_vs_obs_long.csv", index=False)
    met = pd.DataFrame(metrics)
    met.to_csv(outdir / "comparison_metrics.csv", index=False)

    wide_parts = []
    for well, sub in long.groupby("well"):
        tmp = sub[["Date", "time_days_from_20190113", "observed_delta_h_m", "simulated_delta_h_m", "residual_sim_minus_obs_m"]].copy()
        tmp = tmp.rename(
            columns={
                "observed_delta_h_m": f"{well}_obs_delta_h_m",
                "simulated_delta_h_m": f"{well}_sim_delta_h_m",
                "residual_sim_minus_obs_m": f"{well}_residual_m",
            }
        )
        wide_parts.append(tmp)
    wide = wide_parts[0]
    for tmp in wide_parts[1:]:
        wide = pd.merge(wide, tmp, on=["Date", "time_days_from_20190113"], how="outer")
    wide = wide.sort_values("time_days_from_20190113")
    wide.to_csv(outdir / "comparison_sim_vs_obs_wide.csv", index=False)

    plt.figure(figsize=(11, 6))
    for well, sub in long.groupby("well"):
        sub = sub.sort_values("time_days_from_20190113")
        plt.plot(
            sub["time_days_from_20190113"],
            sub["observed_delta_h_m"],
            marker="o",
            linewidth=1.8,
            label=f"{well} obs",
        )
        plt.plot(
            sub["time_days_from_20190113"],
            sub["simulated_delta_h_m"],
            marker="s",
            markersize=3,
            linewidth=1.4,
            linestyle="--",
            label=f"{well} sim",
        )
    plt.axhline(0, linewidth=0.8)
    plt.axvline(float(long["time_days_from_20190113"].min()), linewidth=0.8, linestyle=":")
    plt.xlabel("Time from 2019-01-13 (days)")
    plt.ylabel("Head change dh (m)")
    plt.title("Observed vs simulated monitoring-well head change after 2019-01-13")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "comparison_head_change_all_wells.png", dpi=300)
    plt.close()

    for well, sub in long.groupby("well"):
        sub = sub.sort_values("time_days_from_20190113")
        plt.figure(figsize=(9, 5.5))
        plt.plot(
            sub["time_days_from_20190113"],
            sub["observed_delta_h_m"],
            marker="o",
            linewidth=2.0,
            label=f"{well} observed",
        )
        plt.plot(
            sub["time_days_from_20190113"],
            sub["simulated_delta_h_m"],
            marker="s",
            markersize=3,
            linewidth=1.3,
            linestyle="--",
            label=f"{well} simulated",
        )
        plt.axhline(0, linewidth=0.8)
        plt.axvline(float(sub["time_days_from_20190113"].min()), linewidth=0.8, linestyle=":")
        plt.xlabel("Time from 2019-01-13 (days)")
        plt.ylabel(f"Head change dh (m), zeroed at day {float(sub['time_days_from_20190113'].min()):g}")
        plt.title(f"{well}: simulated vs observed head change after 2019-01-13")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", well)
        plt.savefig(outdir / f"comparison_head_change_{safe}.png", dpi=300)
        plt.close()

    print("Done.")
    print(f"Output directory: {outdir.resolve()}")
    print(met.to_string(index=False))


if __name__ == "__main__":
    main()
