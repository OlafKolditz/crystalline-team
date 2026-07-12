#!/usr/bin/env bash
set -euo pipefail

python compare_simulated_observed_head_change.py \
  --pvd output/case1_k9_2e-11_k10_4e-13_200d.pvd \
  --monitor_csv monitoring/recommended_monitoring_points_all7_ZK501_bedrock.csv \
  --obs_xlsx monitoring/yangyi_after_20190113_for_ogs.xlsx \
  --obs_sheet Monitor_compare_after20190113 \
  --out_dir comparison_case1_k9_2e-11_k10_4e-13_200d
