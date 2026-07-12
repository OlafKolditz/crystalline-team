#!/usr/bin/env bash
set -euo pipefail
python extract_monitoring_head_change_2d.py --results output/case2_k9_k10_2e-10_200d.pvd --monitoring monitoring/monitoring_nodes_2d_fractures_filtered.csv --outdir head_change_output
python compare_simulated_observed_head_change_after20190113.py --sim-heads head_change_output/simulated_head_timeseries.csv --obs-xlsx monitoring/yangyi_after_20190113_for_ogs.xlsx --outdir comparison_output
