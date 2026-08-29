#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for model in DFNM DFNM-LRZ DFN DFN-LRZ; do
  log="$root/$model/output/ogs.log"
  count=$(find "$root/$model/output" -maxdepth 1 -name '*.vtu' 2>/dev/null | wc -l)
  if [[ -f "$log" ]] && grep -q 'Simulation completed' "$log"; then
    accepted=$(grep -A1 'whole computation of the time stepping took' "$log" | tail -n 1 | xargs)
    echo "$model: completed; $count VTU files; $accepted"
  elif [[ -f "$log" ]]; then
    step=$(grep 'Time step #[0-9]* started' "$log" | tail -n 1 | sed -E 's/.*Time step #([0-9]+).*/\1/')
    echo "$model: running at step ${step:-initializing}; $count VTU files"
  else
    echo "$model: not started"
  fi
done
