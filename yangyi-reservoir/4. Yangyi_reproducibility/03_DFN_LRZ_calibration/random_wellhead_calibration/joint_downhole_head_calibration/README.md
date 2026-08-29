# Joint downhole-pressure and monitoring-head calibration

This sweep starts from the refined active-well calibration winner and jointly
scores:

- completion-depth pressure at injection well ZK403 and production wells
  ZK208 and ZK203;
- hydraulic-head variation at SC211, ZK207, and ZK206.

The dimensionless objective gives equal total weight to the two data groups
and uses fixed scales based on observed variability:

`sqrt(0.5 * mean((pressure RMSE / pressure scale)^2) + 0.5 * mean((head RMSE / head scale)^2))`

The pressure scales are 0.250, 0.269, and 0.200 MPa for ZK403, ZK208, and
ZK203. The head scales are 8.235, 6.892, and 4.143 m for SC211, ZK207, and
ZK206. Cases satisfying an absolute-pressure RMSE guardrail of 0.5 MPa at
every active well rank ahead of cases that violate it.

Within each group, all three wells have equal weight. For ZK207 and ZK206, the
simulated head used by the objective is the mean of their two valid mapped
monitoring points. Head changes are re-zeroed at simulation day 1, matching the
observation workbook.

Four targeted material permeabilities are varied log-uniformly with a Latin
hypercube design: Materials 1, 5, 12, and 9. Historical sensitivity runs show
that Materials 4 and 10 have negligible leverage on the ZK206 mismatch, so
they remain fixed along with all other project settings and material
properties from `../refined_sweep/best_calibrated.prj`.

Run 36 sampled candidates, the prior winner, and one monitoring-informed seed:

```bash
/home/zhai/ogs-env/bin/python run_joint_calibration.py --workers 4
```

Use `--resume` to rebuild the ranking from completed cases without rerunning
them.

## Completed sweep

All 38 cases completed successfully. The selected winner is `joint_032`, with
these permeabilities:

| Material | Permeability (m2) |
|---|---:|
| 1 | 9.554224501857e-9 |
| 5 | 1.013370361393e-10 |
| 12 | 2.887007451807e-12 |
| 9 | 3.883308202028e-11 |

Its joint objective is 1.3653, compared with 1.6629 for the prior
pressure-only winner. All three active-well pressure RMSEs remain below the
0.5 MPa guardrail.

| Target | Prior RMSE | Joint-winner RMSE |
|---|---:|---:|
| ZK403 pressure | 0.264 MPa | 0.277 MPa |
| ZK208 pressure | 0.446 MPa | 0.440 MPa |
| ZK203 pressure | 0.168 MPa | 0.299 MPa |
| SC211 head change | 15.589 m | 6.893 m |
| ZK207 head change | 17.442 m | 9.832 m |
| ZK206 head change | 5.892 m | 6.303 m |

The sweep markedly improves SC211 and ZK207 while preserving acceptable
active-well pressure errors. ZK206 does not improve: its observed transient
drawdown is not reproduced, consistent with earlier tests showing negligible
sensitivity to Materials 4 and 10. This likely requires a structural or
storage/source-term change rather than another permeability-only adjustment.
