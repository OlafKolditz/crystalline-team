#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ogs_bin="${OGS_BIN:-/home/zhai/ogs-env/bin/ogs}"

cd "$root"
/home/zhai/ogs-env/bin/python build_cases.py

run_case() {
  local folder="$1"
  local project="$2"
  mkdir -p "$root/$folder/output"
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" "$ogs_bin" "$root/$folder/$project" \
    -o "$root/$folder/output" >"$root/$folder/output/ogs.log" 2>&1
}

run_case DFNM bestfit_DFNM_200d.prj &
pid_dfnm=$!
run_case DFNM-LRZ bestfit_DFNM_LRZ_200d.prj &
pid_lrz=$!
run_case DFN bestfit_DFN_200d.prj &
pid_dfn=$!
run_case DFN-LRZ bestfit_DFNM_LRZ_200d.prj &
pid_dfn_lrz=$!

status=0
wait "$pid_dfnm" || status=1
wait "$pid_lrz" || status=1
wait "$pid_dfn" || status=1
wait "$pid_dfn_lrz" || status=1
exit "$status"
