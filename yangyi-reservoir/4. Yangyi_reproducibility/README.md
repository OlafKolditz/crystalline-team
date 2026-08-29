# Yangyi hierarchical model-reduction reproducibility package

## Purpose and scope

This compact package supports a manuscript describing a hierarchical model-reduction framework for a fault-controlled geothermal reservoir, using the Yangyi geothermal field as the application. It contains source projects, input meshes, observation/forcing data, and scripts needed for the manuscript-relevant calculations. Precomputed OGS transient fields, rankings derived from those fields, figures, caches, and logs are deliberately excluded.

The scientific workflow represented here is:

```text
common 3-D structural model
  -> Full DFNM hydraulic exploration/reference
  -> reduced DFN-LRZ hydraulic calibration
  -> four-model structural comparison
  -> hydraulically supported topology
  -> effective four-pipe tracer calibration
  -> auxiliary long-term thermal consistency assessment
```

The Full DFNM reference and DFN-LRZ calibration are distinct hydraulic stages. The four-model comparison then transfers the selected DFN-LRZ `joint_007` permeability set to all structural configurations; it is a transfer/validation comparison, not four independent recalibrations.

## Software

- OpenGeoSys (OGS): runs all `.prj` files. The exact version is not recorded in the supplied sources.
- PorePy: constructs fracture geometry/topology and intersections. Exact version not recorded.
- Gmsh: creates the conforming mixed-dimensional mesh. The retained file uses Gmsh MSH format 4.1, which is a file-format version rather than evidence of the executable version.
- Python: sweep generation, mesh conversion, scoring, and plotting. Exact interpreter version not recorded. Imported packages include NumPy, pandas, SciPy, matplotlib, meshio, PyVista, and openpyxl/xlrd as applicable.
- ParaView/PyVista-compatible VTK tooling: field extraction and visualization. No ParaView version is recorded.

Do not infer software versions from file timestamps. Create a pinned environment before archival if bitwise software provenance is required.

## Geometry and meshing

The supplied executable workflow supports:

```text
PorePy -> fracture geometry/topology and intersection handling
        -> Gmsh -> conforming mixed-dimensional mesh
        -> conversion/material assignment -> OGS-compatible meshes
```

Verified domain bounds are `x = 243300–247300 m`, `y = 3289500–3293500 m`, and `z = 1084–5084 m` (4 km in each direction). The retained unified mesh has 35,808 nodes, 201,734 tetrahedra, 32,369 2-D fault triangles, and 647 1-D fracture-intersection line elements. The executable generator specifies characteristic sizes of 200 m at the boundary, 80 m on fractures, and 20 m minimum. See `01_geometry_mesh/README.md`.

## Repository structure

- `01_geometry_mesh/`: raw fault/well geometry, meshing scripts, Gmsh intermediates, mixed-dimensional and OGS input meshes.
- `02_full_DFNM_reference/`: the only Full-DFNM project in the supplied `case1`, including matrix, faults, LRZ, forcing, boundary meshes, and monitoring data.
- `03_DFN_LRZ_calibration/`: baseline and systematic coarse, refined, joint, and head-refinement input projects; final `joint_007`; design tables and generators.
- `04_four_model_comparison/`: active DFNM-noLRZ, Full DFNM, Surface DFN, and DFN-LRZ projects with transferred `joint_007` permeabilities.
- `05_tracer_calibration_sweeps/`: 16 LHS and four anchor input projects; selected `anchor_80C` and tracer observations.
- `06_thermal_consistency/`: nine local M203-return-temperature/dispersivity refinement projects; selected 190 °C/35 m case.
- `07_field_observations/`: canonical copies of hydraulic and tracer observations and operational forcing. The thermal workbook was not present in the supplied thermal directory.
- `08_postprocessing_scripts/`: discoverability copies of meshing, calibration, comparison, scoring, and plotting scripts. Run case-local originals where relative paths are assumed.
- `CASE_INDEX.csv`: manuscript-case registry.
- `DEPENDENCY_AUDIT.csv`: project and script dependency/portability findings.
- `FILE_MANIFEST.csv`: SHA-256 file manifest.
- `AUDIT_REPORT.md`: final inclusion/exclusion and reproducibility audit.

## Exact manuscript cases

| Workflow stage | Repository case | Main project | Required data | Figure identifier |
|---|---|---|---|---|
| Full DFNM reference | `02_full_DFNM_reference` | `yangyi_actual_ops_after20190113_outlier_checked.prj` | local mesh/source/monitor files; hydraulic workbook | not specified in sources |
| DFN-LRZ selected calibration | `03_DFN_LRZ_calibration/.../head_refinement` | `projects/joint_007.prj` | category-root mesh/source/monitor files and hydraulic workbooks | not specified |
| Four structures | `04_four_model_comparison/{DFNM,DFNM-LRZ,DFN,DFN-LRZ}` | each folder's `bestfit_*.prj` | local model inputs and monitoring workbook | not specified |
| Tracer selected case | `05_tracer_calibration_sweeps/runs/anchor_80C` | source `case6_joint_anchor_80C.prj`; run `*_portable.prj` | local pipe mesh; tracer CSV for scoring | not specified |
| Thermal selected case | `06_thermal_consistency/runs/M203T190C_S203A0p3x_S208A6x_alphaL35m` | same-name `.prj` | local pipe mesh; external workbook for evaluation | not specified |

No defensible manuscript section or figure numbers were present in the supplied sources, so the index records descriptive sections and `not specified` rather than inventing numbers.

## Recommended run order

1. If mesh regeneration is required, run `01_geometry_mesh/generate_yangyi_dfn_split_outputs.py` from `01_geometry_mesh/` after installing PorePy, Gmsh, meshio, and PyVista. Otherwise use the retained verified input meshes.
2. Run the Full DFNM project from `02_full_DFNM_reference/` with `ogs yangyi_actual_ops_after20190113_outlier_checked.prj`.
3. For the reduced hydraulic history, run the case-local generators in coarse -> refined -> joint -> head-refinement order. The supplied generators have external helper-path dependencies listed in `DEPENDENCY_AUDIT.csv`; every generated `.prj` can instead be run directly from the category root. `joint_007` is the final best-tested project.
4. Run the four projects with `04_four_model_comparison/run_all.sh`. Do not run `build_cases.py` unless its original external base-case hierarchy has been restored; the final transferred projects are already included.
5. Run individual tracer `*_portable.prj` files from their own run directories. The full joint generator additionally needs three earlier tracer helper branches not supplied here.
6. Run individual thermal `.prj` files from their own run directories. The sweep generator/evaluator additionally needs the parent case-3 project and `Power output and PT.xls`.

OGS outputs should be written into newly created output directories and are intentionally absent from this archive.

## Field observations and roles

- Active hydraulic calibration targets: completion-depth absolute pressure at ZK403, ZK208, and ZK203, transformed from monitored wellhead pressure where appropriate.
- Joint head targets: head change at SC211 and ZK207; ZK206 was a joint-stage target but became diagnostic in the final local head refinement. Other monitors (ZK204, ZK401, ZK402, ZK501) are evaluation/diagnostic series.
- Hydraulic forcing: ZK403 injection and ZK208/ZK203 production histories. Curves are embedded in projects; the operational CSV/workbook provides traceability.
- Tracer calibration: processed 2019 long-test concentrations for 2,6-naphthalenedisulfonic acid disodium salt at ZK208 and ZK203. Manuscript peak targets are 5 d/518.88 ppb and 32 d/164.60 ppb, respectively.
- Thermal evaluation: ZK203 and ZK208 production temperatures and ZK403 injection temperature/rate. Curves needed to run the selected project are embedded, but the original evaluation workbook is an external missing dependency.

Wellhead pressure, completion-depth absolute pressure, and hydraulic-head change are distinct observables and must not be merged or relabelled.

## Calibration results represented by source inputs

`joint_007` is the pressure-feasible best tested case under the recorded sequence, not a demonstrated unique/global optimum. Its active-well RMSE values are 0.289 MPa (ZK403), 0.428 MPa (ZK208), and 0.427 MPa (ZK203). Head RMSE is 6.798 m at SC211 and 3.609 m at ZK207; ZK206 is diagnostic at 7.272 m.

The selected tracer anchor assigns one third of flow to the explicit pipe/storage representation and two thirds to unresolved deep circulation, uses 5 m longitudinal and 0.1 m transverse dispersivity in the fast paths, 70% eventual deep recovery, a 60 d deep release time, and a 238 d simulation. This is an effective transport topology, not four literal fractures and not a partition determined by the hydraulic model.

The selected thermal case uses an effective initial temperature of 185 °C, M203/M208 deep-return temperatures of 190/200 °C, effective solid heat capacities of 240 J kg-1 K-1 on ZK203 paths (0.3x), 4,800 J kg-1 K-1 on ZK208 paths (6x), and 800 J kg-1 K-1 otherwise, plus 35 m longitudinal and 0.1 m transverse thermal dispersivity. It runs 1,461 d. Full-period RMSE is 5.221 °C at ZK203 and 7.437 °C at ZK208 (mean 6.329 °C); the six well/stage balanced RMSE is 8.378 °C.

## Reproducing model-based figures

- Active-well pressure comparison: run all four comparison cases, then execute `plot_well_pressure_comparisons_portable.py` with generated PVD/VTU outputs present.
- SC211/ZK207 monitoring heads: run all four cases, then `plot_monitoring_head_comparisons_portable.py --model all`.
- Pressure-change and Darcy-velocity fields: run all four cases, then `plot_final_field_comparisons.py`. The script slices 3-D matrix results on a constant-y plane through each mesh centre; it does **not** implement a Z-Y plane through the ZK203 depth. The latter interpretation is therefore not supported by this script.
- Hydraulic calibration curves/metrics: run the applicable sweep generator and use its built-in scoring/plotting. Missing external helper modules must first be restored.
- Tracer BTC: run `anchor_80C` (or all joint cases), generate its tracer time-series CSV using the original feedback workflow, then run `plot_best_case_single_axis.py` with the observed tracer CSV.
- Thermal consistency: run the selected thermal project, extract `production_temperature_4y.csv` using `run_joint_refinement.py`, then use `plot_injection_production_temperature.py`; both evaluation steps need the missing thermal workbook.

Generated figures and numerical result tables are not included.

## Scientific cautions and known limitations

- Low resistivity itself is not proof of high permeability.
- The hydraulic comparison constrains effective structural connectivity.
- The hydraulic model does not determine the one-third/two-thirds shallow/deep tracer partition.
- The four-pipe model is an effective reduced transport representation, not four literal fractures.
- Effective thermal heat capacities are lumped parameters, not intrinsic rock heat capacities.
- The reduced thermal calculation is an auxiliary consistency test, not the final reservoir-scale predictive thermal model. Long-term predictive breakthrough should ultimately be evaluated after reintroducing the calibrated topology into the Full DFNM framework.
- The thermal project solves pressure, component transport, and temperature, but the supplied constants do not establish two-way temperature feedback to flow properties; do not call it fully two-way coupled HT.
- Selected cases are best-tested cases under documented procedures, not necessarily unique or global optima.
- Surface DFN versus DFN-LRZ is not a strict one-factor LRZ ablation because the Surface DFN retains only two active wells while DFN-LRZ retains three.
- Faults are simplified/planar; the reduced topology omits distributed fracture/storage processes; parameter non-uniqueness remains.

## Portability

The `.prj` files themselves contain relative mesh references and all required meshes are present. Nested DFN-LRZ projects should be launched with `03_DFN_LRZ_calibration/` as the working directory because their mesh paths are category-root relative. Tracer and thermal projects are self-contained within each run directory.

No project was rewritten. Original scripts retain their original paths and scientific logic. Two clearly named comparison-script copies contain documented path-only changes; no calculation or scientific setting changed. Known external dependencies and context-sensitive discoverability copies are listed in `DEPENDENCY_AUDIT.csv` and `AUDIT_REPORT.md`.
