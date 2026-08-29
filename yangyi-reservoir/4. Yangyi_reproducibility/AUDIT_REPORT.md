# Final repository audit

## Package summary

- Exact path: `/home/zhai/Yangyi_OGS/Yangyi_reproducibility`
- Size at audit: 126,925,357 bytes (about 121 MiB; manifest regeneration may change this slightly)
- Files at audit: 705 (704 entries plus the self-excluded manifest)
- OGS projects: 180, comprising retained source projects plus 20 documented path-only tracer copies
- Mesh/geometry files: 381 (`.vtu`, `.vtp`, `.msh`, `.xao`, `.geo_unrolled`)
- Python/shell files: 49 including case-local originals, categorized discoverability copies, portable comparison copies, and audit utilities
- Project mesh-reference audit: 894 references; zero missing references in manuscript-relevant/retained projects. Nested calibration projects resolve from the documented category-root working directory.

## Included manuscript-relevant cases

1. Full DFNM reference: `02_full_DFNM_reference/yangyi_actual_ops_after20190113_outlier_checked.prj`. It is the only project in the supplied Full `case1`, so no silent selection among candidates was made.
2. DFN-LRZ final: `03_DFN_LRZ_calibration/random_wellhead_calibration/joint_downhole_head_calibration/head_refinement/projects/joint_007.prj`, plus the entire recorded systematic input sequence.
3. Four comparison projects:
   - DFNM-noLRZ: `04_four_model_comparison/DFNM/bestfit_DFNM_200d.prj`
   - Full DFNM: `04_four_model_comparison/DFNM-LRZ/bestfit_DFNM_LRZ_200d.prj`
   - Surface DFN: `04_four_model_comparison/DFN/bestfit_DFN_200d.prj`
   - DFN-LRZ: `04_four_model_comparison/DFN-LRZ/bestfit_DFNM_LRZ_200d.prj`
4. Tracer final source project: `05_tracer_calibration_sweeps/runs/anchor_80C/case6_joint_anchor_80C.prj`; runnable copy: `case6_joint_anchor_80C_portable.prj`. All 16 LHS and four anchor source/portable pairs are retained.
5. Thermal final: `06_thermal_consistency/runs/M203T190C_S203A0p3x_S208A6x_alphaL35m/M203T190C_S203A0p3x_S208A6x_alphaL35m.prj`, plus all nine local-refinement inputs.

The comparison source README, transfer CSV, and project values support a common `joint_007` permeability basis and explicitly state that these four structures were not independently recalibrated.

## Observation and forcing datasets

Canonical copies in `07_field_observations` are:

- `hydraulic/yangyi_after_20190113_for_ogs.xlsx`
- `hydraulic/clean_six_line_pressure_data.xlsx`
- `hydraulic/observed_monitoring_delta_head.csv`
- `hydraulic/ogs_source_curves_after20190113_outlier_checked.csv`
- `tracer/observed_tracer_ZK203_ZK208.csv`

Projects also retain required embedded operational curves and local dependency copies. The original thermal observation/operation workbook was not found in the supplied thermal directory.

## Scripts

Included functions cover PorePy/Gmsh generation and conversion, LRZ/boundary/material mapping, coarse/refined/joint/head hydraulic calibration, pressure/head scoring, four-model construction and pressure/head/field plotting, tracer joint sweep/BTC plotting, and thermal refinement/error plotting. The per-script contract is in `08_postprocessing_scripts/README.md`.

Two original comparison plotters referenced old sibling directories. Their originals are preserved and `*_portable.py` copies change only those data paths. Twenty tracer source projects use absolute original mesh paths; the originals are preserved and verified `*_portable.prj` copies change only the seven top-level mesh paths. An XML comparison after normalizing mesh text verified that all 20 portable copies are otherwise identical.

## Missing external dependencies and portability limits

- DFN-LRZ sweep generators call `compare_observed_wellhead_pressure.py` from a sibling hydraulic base case outside the supplied calibration source.
- `build_cases.py` needs the original full/no-matrix/Surface-DFN base hierarchy; final generated comparison projects are included, so rebuilding them is optional.
- `run_joint_sweep.py` imports `run_deep_feedback.py`, `run_flow_fraction_sweep.py`, and `run_tau_sweep.py` from three earlier tracer branches outside the supplied joint-sweep source.
- Thermal generation/scoring needs parent `case3_split_storage_heat_exchange_4y`, `Power output and PT.xls`, and its `.python_deps` context.
- Several scripts default to `/home/zhai/ogs-env/bin/ogs` or `/home/zhai/ogs-env/bin/python`; use a local OGS/Python environment and `OGS_BIN` where supported.
- Exact software versions are absent.
- Central `08_postprocessing_scripts` copies are an index; relative-path scripts should be run beside their cases.

There are no unresolved mesh references in retained project files when using the documented working directories/portable copies. The 140 absolute references reported by the audit belong to the 20 preserved original tracer projects; their portable counterparts resolve locally.

## Exclusions

Excluded from all six sources: PVD collections, transient/result VTUs, output/result folders, restart/checkpoint material, simulation logs, iteration histories/curves, rankings and derived metric tables, generated pressure/head/tracer/thermal CSV results, PNG/SVG figures, caches, compiled Python files, old archives, and the archived duplicate DFNM-noLRZ project. Diagnostic well-intersection visualization meshes and a duplicate DOCX mesh tutorial were also excluded as unnecessary for rerunning manuscript cases.

Input `.vtu` files were retained only as bulk, pipe, boundary, source, monitoring, or geometry-generation meshes. A prohibited-file scan found no `.pvd`, generated figure, log, transient `_ts_*.vtu`, ranking/metrics result, output directory, results directory, or cache directory.

## Reproducibility status by result

| Major result | Status from package | Qualification |
|---|---|---|
| Common geometry/mesh | Reproducible in principle; verified mesh supplied | exact package versions absent, so bitwise remeshing is not guaranteed |
| Full DFNM simulation | Runnable | field/head outputs must be regenerated |
| Exact DFN-LRZ projects including `joint_007` | Runnable | sweep regeneration/scoring needs one external pressure helper |
| Four-model simulations and pressure/head/field plots | Runnable | use documented portable pressure/head plotters |
| Selected tracer simulation | Runnable via portable project | automated feedback regeneration/scoring and BTC extraction need external earlier-branch helpers |
| Full 20-case tracer design | Projects runnable | exact automated calibration workflow is partial because helpers are missing |
| Selected and nine-case thermal simulations | Runnable | independent scoring/figure reproduction is blocked by the missing workbook |
| Thermal consistency metrics | Documented from source | cannot be recalculated from package alone without observation workbook |

## Final confirmation

Original sources were not modified, renamed, deleted, or overwritten. The package was created by copying filtered scientific inputs and adding new documentation/audit/portable path-only copies. No simulations were rerun. No large precomputed simulation results were copied.
