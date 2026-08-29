# Hydraulic-head comparison for the refined calibration winner

This directory contains the simulated-versus-observed hydraulic-head variation
comparison for `../best_calibrated.prj` (`random_016`). The simulation covers
200 days with daily output. Simulated and observed head changes are re-zeroed
at day 1, the first monitoring observation after 2019-01-13.

The comparison uses
`recommended_monitoring_points_all7_ZK501_bedrock_noMatrix.csv`. This is the
node map matching the 17,112-node no-matrix mesh. Wells with two mapped
monitoring points retain both curves in their individual plots; the combined
plot uses their mean simulated response. The per-well error summary pools the
residuals from both mapped points, so those wells have 64 comparison records
instead of 32.

Reproduce the run and comparison from `refined_sweep/` with:

```bash
bash run_best_head_comparison.sh
```

## Results

OGS completed 200 accepted daily timesteps with no rejected steps. The
simulated-versus-observed head-change errors are:

| Well | Comparison records | MAE (m) | RMSE (m) | Mean bias (m) |
|---|---:|---:|---:|---:|
| SC211 | 32 | 13.898 | 15.589 | 13.898 |
| ZK204 | 32 | 1.299 | 2.115 | -0.884 |
| ZK206 | 64 | 4.388 | 5.892 | 4.237 |
| ZK207 | 64 | 16.508 | 17.443 | 16.508 |
| ZK401 | 64 | 22.950 | 23.907 | 22.950 |
| ZK402 | 64 | 1.245 | 1.278 | 1.245 |
| ZK501 | 32 | 10.113 | 10.604 | 10.113 |

Positive bias means that the simulated head change is higher than observed;
negative bias means it is lower.
