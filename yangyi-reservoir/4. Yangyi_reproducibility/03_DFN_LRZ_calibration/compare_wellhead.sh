#!/usr/bin/env bash
set -euo pipefail

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
shared_dir="$case_dir/../case1_k9_2e-11_k10_4e-13"
python_bin="$case_dir/../.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin=python
fi
export MPLCONFIGDIR="${TMPDIR:-/tmp}/yangyi-matplotlib"
mkdir -p "$MPLCONFIGDIR"

"$python_bin" "$shared_dir/plot_wellhead_pressure.py" \
  --case-dir "$case_dir" \
  --pvd "$case_dir/output/case1_k9_2e-11_k10_4e-13_200d.pvd" \
  --out-dir "$case_dir/wellhead_pressure"

"$python_bin" "$shared_dir/compare_observed_wellhead_pressure.py" \
  --workbook "$case_dir/monitoring/yangyi_after_20190113_for_ogs.xlsx" \
  --simulation "$case_dir/wellhead_pressure/wellhead_pressure_timeseries.csv" \
  --out-dir "$case_dir/wellhead_pressure/comparison_observed"
