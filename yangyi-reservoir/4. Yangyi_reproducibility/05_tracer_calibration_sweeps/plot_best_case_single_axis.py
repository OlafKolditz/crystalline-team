#!/usr/bin/env python3
"""Plot the best four-pipe tracer calibration on a single axis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "anchor_80C"
OBSERVED = RUN / "observed_tracer_ZK203_ZK208.csv"
SIMULATED = RUN / "tracer_timeseries_ppb.csv"
OUTPUT_STEM = HERE / "best_case_tracer_calibration_single_axis"

COLORS = {
    "ZK208": "#d62728",  # red
    "ZK203": "#1f77b4",  # blue
}


def main() -> None:
    observed = pd.read_csv(OBSERVED, encoding="utf-8-sig")
    simulated = pd.read_csv(SIMULATED)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)

    for well in ("ZK208", "ZK203"):
        color = COLORS[well]
        obs = observed.loc[observed["well"] == well].sort_values(
            "days_since_tracer_injection"
        )
        sim = simulated.loc[simulated["well"] == well].sort_values("time_days")

        ax.plot(
            obs["days_since_tracer_injection"],
            obs["concentration_ppb"],
            color=color,
            marker="o",
            markersize=4.2,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.45,
            linewidth=1.15,
            label=f"{well} observed",
            zorder=3,
        )
        ax.plot(
            sim["time_days"],
            sim["concentration_ppb"],
            color=color,
            linestyle="--",
            linewidth=1.8,
            label=f"{well} simulated",
            zorder=2,
        )

    ax.set_xlim(0, 238)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time since tracer injection (days)")
    ax.set_ylabel("Tracer concentration (ppb)")
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.65)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    for suffix in ("png", "svg", "pdf"):
        options = {"dpi": 400} if suffix == "png" else {}
        fig.savefig(OUTPUT_STEM.with_suffix(f".{suffix}"), **options)
    plt.close(fig)


if __name__ == "__main__":
    main()
