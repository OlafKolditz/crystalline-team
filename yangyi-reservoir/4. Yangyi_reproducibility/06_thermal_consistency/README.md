# case4_local_joint_refinement_4y

Local four-year joint refinement based on the best case from
`case3_split_storage_heat_exchange_4y`. The real 2019--2022 ZK403 injection
temperature and rate, hydraulic parameters, storage factors, initial
temperature, and M208 deep-return temperature are unchanged.

## Parameter grid

- M203 deep-return temperature: 190, 192, 194 degC
- Thermal longitudinal dispersivity: 25, 30, 35 m
- Total: nine simulations

The completed case3 centre point (194 degC, 30 m) is copied and reused. All
other combinations are newly simulated.

## Evaluation windows

- Initial response: day 12 to day 120
- Cooling front: day 120 to day 240
- Long-term response: day 240 to day 1461

Cases are ranked first by the equally weighted mean of the six well/stage
RMSE values, then by the conventional full-period mean RMSE. This prevents
the much longer late-time record from hiding early-front mismatch.

## Completed result

All nine cases completed successfully. The ranking is:

| rank | M203 (degC) | alphaL (m) | full mean RMSE (degC) | stage-balanced RMSE (degC) |
|---:|---:|---:|---:|---:|
| 1 | 190 | 35 | 6.329 | 8.378 |
| 2 | 192 | 35 | 6.457 | 8.414 |
| 3 | 194 | 35 | 6.748 | 8.530 |
| 4 | 194 | 30 | 6.901 | 8.852 |
| 5 | 192 | 30 | 6.867 | 8.858 |
| 6 | 190 | 30 | 7.002 | 8.940 |
| 7 | 194 | 25 | 7.859 | 9.606 |
| 8 | 192 | 25 | 8.086 | 9.728 |
| 9 | 190 | 25 | 8.426 | 9.893 |

The recommended tested case is
`runs/M203T190C_S203A0p3x_S208A6x_alphaL35m`. Its full-period biases are
+0.275 degC at ZK203 and -0.406 degC at ZK208. The optimum alphaL is at the
35 m upper boundary, so this result identifies a possible narrow follow-up
search direction around 35--40 m; it is not evidence for increasing alphaL
without limit.

```bash
/home/zhai/ogs-env/bin/python -u run_joint_refinement.py
/home/zhai/ogs-env/bin/python plot_injection_production_temperature.py
```

`injection_production_temperature_comparison_4y.png` combines the three-well
temperature histories in one panel. ZK208 is red, ZK203 orange, and ZK403
green; observations are dots and simulations are lines. For ZK403, the line
is the temperature boundary applied to the simulation and is hidden when the
injection rate is zero.
