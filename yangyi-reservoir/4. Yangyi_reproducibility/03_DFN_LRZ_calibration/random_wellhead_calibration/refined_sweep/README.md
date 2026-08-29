# Refined wellhead-pressure permeability sweep

This is the second calibration round. Its `baseline_000` is the first-round
winner (`random_016`), whose weighted objective was `0.4981 MPa`.

| Parameter | Well | First-round winner | Refined random range |
| --- | --- | ---: | ---: |
| Material 5 permeability (`k5`) | ZK403 | `1.593e-10 m2` | `5e-11`–`2e-10 m2` |
| Material 1 permeability (`k1`) | ZK208 | `3.205e-9 m2` | `7e-9`–`3e-8 m2` |
| Material 12 permeability (`k12`) | ZK203 | `3.000e-12 m2` | `4e-12`–`2e-11 m2` |

The objective remains:

`0.60 * RMSE(ZK403) + 0.20 * RMSE(ZK208) + 0.20 * RMSE(ZK203)`.

Run 24 refined random trials plus the first-round winner with:

```bash
/home/zhai/ogs-env/bin/python run_refined_calibration.py
```

Use `--resume` to regenerate rankings and best-case outputs without rerunning
successful OGS simulations.
