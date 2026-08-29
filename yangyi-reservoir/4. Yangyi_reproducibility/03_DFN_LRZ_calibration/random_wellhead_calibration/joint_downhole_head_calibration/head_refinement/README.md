# Pressure-constrained SC211/ZK207 head refinement

This local sweep starts from `../best_joint_calibrated.prj`. Materials 1 and 5
are fixed at the joint winner so the ZK208 and ZK403 completion-pressure fits
are retained. Only Materials 12 and 9 are refined because they control SC211
and the two mapped ZK207 responses.

Cases are feasible only when their absolute completion-pressure RMSE satisfies:

- ZK403: at most 0.32 MPa
- ZK208: at most 0.46 MPa
- ZK203: at most 0.45 MPa

Feasible cases are ranked by the equally weighted, observed-scale-normalized
RMSE of SC211 and ZK207. ZK206 remains a diagnostic output but is not part of
this refinement objective because prior sensitivity tests found negligible
leverage from its mapped Materials 4 and 10.

Run 30 Latin-hypercube candidates plus the current winner and an informed seed:

```bash
/home/zhai/ogs-env/bin/python run_head_refinement.py --cases 30 --workers 4
```

## Completed refinement

All 32 cases completed successfully. The selected pressure-feasible winner is
`joint_007`. Materials 1 and 5 remain fixed at the previous joint winner, and
the refined values are:

| Material | Permeability (m2) |
|---|---:|
| 1 (fixed) | 9.554224501857e-9 |
| 5 (fixed) | 1.013370361393e-10 |
| 12 | 2.499956611737e-12 |
| 9 | 1.125163238659e-11 |

| Target | Previous joint winner | Head-refinement winner |
|---|---:|---:|
| ZK403 pressure RMSE | 0.277 MPa | 0.289 MPa |
| ZK208 pressure RMSE | 0.440 MPa | 0.428 MPa |
| ZK203 pressure RMSE | 0.299 MPa | 0.427 MPa |
| SC211 head RMSE | 6.893 m | 6.798 m |
| ZK207 head RMSE | 9.832 m | 3.609 m |
| ZK206 head RMSE (diagnostic) | 6.303 m | 7.272 m |

The refinement preserves the requested active-well pressure limits and
substantially improves ZK207. SC211 improves slightly. The remaining conflict
is controlled by Material 12: reducing it further improves drawdown but pushes
the ZK203 pressure error beyond the accepted limit.
