#!/usr/bin/env bash
set -euo pipefail
mkdir -p output
ogs c2_k9_k10_2e-10_200d.prj -o output 2>&1 | tee output/case2_2D.log
