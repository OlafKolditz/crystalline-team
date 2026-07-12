#!/usr/bin/env bash
set -euo pipefail
mkdir -p output
ogs c1_k9_2e-11_k10_4e-13_200d.prj -o output 2>&1 | tee output/case1_2D.log
