#!/usr/bin/env bash
set -euo pipefail

python compare_simulated_observed_head_change.py \
  --pvd output/case2_k9_k10_2e-10_200d.pvd \
  --monitor_csv monitoring/recommended_monitoring_points_all7_ZK501_bedrock.csv \
  --obs_xlsx monitoring/yangyi_after_20190113_for_ogs.xlsx \
  --obs_sheet Monitor_compare_after20190113 \
  --out_dir comparison_case2_k9_k10_2e-10_200d
