# Best-fit four-model liquid-flow comparison

This package transfers the permeability values from the calibrated
`joint_007.prj` into four requested base models without changing their other
physical, forcing, temporal, or numerical settings.

| Folder | Base project | Geometry | Wells retained |
|---|---|---|---|
| `DFNM/` | full base project | Matrix + fractures; no Material 12 LRZ | ZK403, ZK208, ZK203 |
| `DFNM-LRZ/` | full base project | Matrix + fractures + Material 12 LRZ | ZK403, ZK208, ZK203 |
| `DFN/` | `test-2D/case1_3D_initial_pressure/...200d.prj` | Fractures only | ZK403 injection and ZK208 production |
| `DFN-LRZ/` | calibrated no-matrix base project | Fractures + Material 12 LRZ; no Material 0 | ZK403, ZK208, ZK203 |

DFNM contains 201,734 Material 0 matrix cells and no Material 12 cells.
DFNM-LRZ contains 198,046 Material 0 matrix cells and 3,688 Material 12 LRZ
cells. DFN contains only fracture/fault materials 1--10. DFN-LRZ contains
fracture/fault materials 1--10 plus the 3,688 Material 12 LRZ cells and no
Material 0 cells.
The DFN retains the requested two-well, exactly balanced forcing and therefore
does not include ZK203. This forcing difference must be acknowledged when
comparing DFN with the two full models.

Transferred permeabilities (m2):

| Material ID | Permeability |
|---:|---:|
| 0 | `1.0e-16` |
| 1 | `9.554224501857e-09` |
| 2 | `4.0e-17` |
| 3 | `4.0e-17` |
| 4 | `4.0e-17` |
| 5 | `1.013370361393e-10` |
| 6 | `4.0e-10` |
| 7 | `4.0e-09` |
| 8 | `4.0e-10` |
| 9 | `1.125163238659e-11` |
| 10 | `4.0e-13` |
| 12 | `2.499956611737e-12` |

Material 12 is active in DFNM-LRZ and DFN-LRZ and unused in DFNM and DFN.
Material 0 is active in DFNM and DFNM-LRZ only. The unused DFN base definitions for Materials 11 and
100 are retained because `joint_007` does not define them.

The full models retain 200 daily steps. The requested DFN base retains its
original 199 daily steps (day 0 through day 199). Pressure and Darcy velocity
are output by every project.

Run all four concurrently with:

```bash
./run_all.sh
```

The exact transferred values are recorded in `permeability_transfer.csv`, and
`source_mapping_audit.csv` verifies that every source mesh carries the correct
bulk-node ID for its copied bulk mesh.

## Active-well pressure figures

After the simulations finish, generate the requested figures with:

```bash
/home/zhai/ogs-env/bin/python plot_well_pressure_comparisons.py
```

Each model receives an injection-well figure and a production-well figure in
its `pressure_comparison/` directory. Simulation is dashed and monitoring data
is a dot-plus-line series. ZK403 is green, ZK208 red, and ZK203 orange. The DFN
production figure contains only ZK208. The plotted data and pressure-error
metrics are saved alongside the figures as CSV files.

Additional `*_completion_pressure.png` figures explicitly include static
water pressure at each completion depth. They compare the OGS completion-node
pressure with monitored wellhead pressure plus `rho*g*depth`. This is
mathematically equivalent to the wellhead comparison but makes the well-depth
correction visible in the plotted pressure values and labels.

## Monitoring-well hydraulic-head figures

Generate the seven-well hydraulic-head-change comparisons with:

```bash
/home/zhai/ogs-env/bin/python plot_monitoring_head_comparisons.py --model all
```

Observed head changes are solid dot-plus-line series and simulations are
dashed. Each monitoring well keeps the same color in every model. Individual
model figures and CSV data are written to each model's `head_comparison/`
directory. The comparison also includes the fourth case `DFN-LRZ`. The
root-level combined figure and
`all_models_head_error_metrics.csv` contain all four cases and RMSE in metres.
Head changes are zeroed at the first observation (day 1), matching the
monitoring workbook.
