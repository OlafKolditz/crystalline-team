#!/usr/bin/env bash
set -euo pipefail

refined_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case_dir="$(cd "$refined_dir/../.." && pwd)"
analysis_dir="$refined_dir/best_hydraulic_head_comparison"
output_dir="$analysis_dir/output"
python_bin=/home/zhai/ogs-env/bin/python
ogs_bin=/home/zhai/ogs-env/bin/ogs

export OMP_NUM_THREADS=2
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-best-calibrated-head"
mkdir -p "$output_dir" "$MPLCONFIGDIR"

"$ogs_bin" "$refined_dir/best_calibrated.prj" \
  -m "$case_dir" \
  -o "$output_dir" \
  2>&1 | tee "$output_dir/ogs.log"

"$python_bin" "$case_dir/compare_simulated_observed_head_change.py" \
  --pvd "$output_dir/random_016.pvd" \
  --monitor_csv "$case_dir/monitoring/recommended_monitoring_points_all7_ZK501_bedrock_noMatrix.csv" \
  --obs_xlsx "$case_dir/monitoring/yangyi_after_20190113_for_ogs.xlsx" \
  --obs_sheet Monitor_compare_after20190113 \
  --out_dir "$analysis_dir"

