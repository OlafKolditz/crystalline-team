#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
marker="$root/DFNM/output/RUN_COMPLETE"

for _ in $(seq 1 480); do
  if [[ -f "$marker" ]]; then
    cd "$root"
    MPLCONFIGDIR=/tmp/bestfit_head_mpl \
      /home/zhai/ogs-env/bin/python plot_monitoring_head_comparisons.py --model all \
      > head_comparison_finalize.log 2>&1
    touch HEAD_COMPARISON_COMPLETE
    exit 0
  fi
  sleep 30
done

echo "Timed out waiting for the revised DFNM simulation." >&2
exit 1
