#!/usr/bin/env python3
"""Plot observed and simulated injection/production temperatures together."""

from __future__ import annotations

import os
from pathlib import Path
import sys

CASE_DIR = Path(__file__).resolve().parent
PARENT = CASE_DIR.parent / "case3_split_storage_heat_exchange_4y"
sys.path.insert(0, str(PARENT / ".python_deps"))
os.environ.setdefault("MPLCONFIGDIR", str(CASE_DIR / ".matplotlib"))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


START_DATE = pd.Timestamp("2019-01-01")
BEST_NAME = "M203T190C_S203A0p3x_S208A6x_alphaL35m"
SIMULATION_CSV = CASE_DIR / "runs" / BEST_NAME / "production_temperature_4y.csv"
BOUNDARY_CSV = PARENT / "ZK403_real_boundary_2019_2022.csv"
EXCEL = PARENT / "Power output and PT.xls"
OUTPUT = CASE_DIR / "injection_production_temperature_comparison_4y.png"

COLORS = {
    "ZK208": "#d62728",  # red
    "ZK203": "#ff7f0e",  # orange
    "ZK403": "#2ca02c",  # green
}


def load_production_observations() -> pd.DataFrame:
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
    for well in ("ZK203", "ZK208"):
        column = f"{well}_observed_C"
        observed[column] = pd.to_numeric(observed[column], errors="coerce")
        observed.loc[~observed[column].between(0.0, 250.0), column] = pd.NA
    return observed.loc[observed["time_days"].between(12.0, 1461.0)].copy()


def main() -> None:
    simulated = pd.read_csv(SIMULATION_CSV, parse_dates=["date"])
    observed = load_production_observations()
    injection = pd.read_csv(BOUNDARY_CSV, parse_dates=["date"])

    fig, axis = plt.subplots(figsize=(13, 7.2))

    # Plot lines first so the observation markers remain visible on top.
    for well in ("ZK208", "ZK203"):
        axis.plot(
            simulated["date"], simulated[f"{well}_temperature_C"],
            color=COLORS[well], lw=2.0, label=f"{well} simulation", zorder=2,
        )
    flowing = injection["injection_rate_t_h"] > 0.0
    axis.plot(
        injection["date"], injection["injection_temperature_C"].where(flowing),
        color=COLORS["ZK403"], lw=1.8, label="ZK403 simulation", zorder=2,
    )

    for well in ("ZK208", "ZK203"):
        axis.scatter(
            observed["date"], observed[f"{well}_observed_C"],
            s=13, color=COLORS[well], alpha=0.60, edgecolors="none",
            label=f"{well} observation", zorder=3,
        )
    injection_observed = injection["temperature_was_observed"].fillna(False).astype(bool)
    axis.scatter(
        injection.loc[injection_observed, "date"],
        injection.loc[injection_observed, "injection_temperature_C"],
        s=13, color=COLORS["ZK403"], alpha=0.60, edgecolors="none",
        label="ZK403 observation", zorder=3,
    )

    axis.set_title("Injection and production temperature: observations and simulation")
    axis.set_xlabel("Date")
    axis.set_ylabel("Temperature (degC)")
    axis.set_xlim(pd.Timestamp("2019-01-01"), pd.Timestamp("2023-01-01"))
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=9, frameon=True)
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
