# Random wellhead-pressure permeability calibration

This sweep targets the materials containing the three active-well completion
nodes in the no-matrix model:

| Parameter | Well | Baseline | Random range |
| --- | --- | ---: | ---: |
| Material 5 permeability (`k5`) | ZK403 injection | `1e-8 m2` | `1e-10`–`3e-9 m2` |
| Material 1 permeability (`k1`) | ZK208 production | `1e-8 m2` | `3e-9`–`3e-8 m2` |
| Material 12 permeability (`k12`) | ZK203 production | `2.5e-12 m2` | `1e-12`–`1e-11 m2` |

Values are sampled log-uniformly using a fixed random seed. The existing
permeability case is included as `baseline_000`.

The ranking objective is the weighted absolute-pressure RMSE:

`0.60 * RMSE(ZK403) + 0.20 * RMSE(ZK208) + 0.20 * RMSE(ZK203)`.

This prioritizes the large ZK403 mismatch while retaining both production
wells in the objective. Per-well absolute and pressure-change RMSE values are
also recorded so that a low aggregate score cannot hide a degraded producer.

Run the default 24 random trials plus the baseline with four OGS workers:

```bash
/home/zhai/ogs-env/bin/python run_random_calibration.py
```

Use `--cases`, `--workers`, and `--seed` to change the sweep. Successful runs
can be reused by rerunning with `--resume`; use `--force` to replace them.
Full VTU output is kept only in a case-specific temporary directory and removed
after well pressures are extracted.

Main outputs are `ranking_all_cases.csv`, `top_10_cases.csv`, per-trial results
under `results/`, and publication-style plots and tables under
`best_case_comparison/`. The winning project is copied to
`best_calibrated.prj`, with its parameters and scores in
`best_case_parameters.csv`.

The second, narrower calibration campaign is under `refined_sweep/`. It uses
the first-round winner as its baseline and targets the parameter regions favored
by the first sweep.
