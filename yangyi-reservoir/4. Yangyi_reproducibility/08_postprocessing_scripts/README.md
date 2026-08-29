# Script inventory

These are exact discoverability copies. Prefer the copies beside each case because many scripts resolve inputs relative to `__file__`. No output required by plotting/post-processing is included; first run the stated simulations.

| Group/scripts | Function | Expected inputs / prerequisite |
|---|---|---|
| `geometry/generate_yangyi_dfn_split_outputs.py` | PorePy/Gmsh construction and VTK conversion | fault CSV/VTP inputs; PorePy, Gmsh, meshio, PyVista |
| `geometry/assign_lowres_materialid.py`, `extract_six_faces_vtu.py` | LRZ material assignment and boundary extraction | generated bulk mesh |
| `geometry/assign_well_cells_material_intersections_vtu.py`, `estimate_fault_parameters_from_vtp.py` | diagnostic/source geometry mapping | mesh plus well/fault geometry |
| `full_DFNM/compare_simulated_observed_head_change_after20190113.py` | monitoring-head extraction and RMSE | completed Full DFNM PVD/VTUs, monitor CSV, hydraulic observations |
| `DFN_LRZ_calibration/run_random_calibration.py` | coarse pressure sweep and scoring | base project, meshes, external pressure helper, OGS |
| `run_refined_calibration.py` | refined pressure sweep | preceding coarse winner/helper context |
| `run_joint_calibration.py` | joint pressure/head LHS sweep | refined winner, pressure/head helpers, observations, OGS |
| `run_head_refinement.py` | pressure-constrained SC211/ZK207 refinement | joint winner and joint helper module |
| `compare_simulated_observed_head_change.py` | head extraction/comparison | completed PVD/VTUs, monitor map, workbook |
| `four_model_comparison/build_cases.py` | generates transferred-parameter projects | original external base-case hierarchy; not needed for retained final projects |
| `plot_well_pressure_comparisons.py` | active-well pressure panels/metrics | all four completed model outputs and hydraulic workbook |
| `plot_monitoring_head_comparisons.py` | SC211/ZK207 and other monitor head panels/metrics | all four completed outputs and workbook |
| `plot_final_field_comparisons.py` | pressure-change and Darcy-velocity spatial panels | initial/final PVD-referenced VTUs for all four models |
| `tracer/run_joint_sweep.py` | fixed-seed LHS/anchor generation, feedback runs, scoring | three external earlier-branch helper scripts, observations, OGS |
| `tracer/plot_best_case_single_axis.py` | selected-case BTC comparison | generated `tracer_timeseries_ppb.csv` and observed tracer CSV |
| `thermal/run_joint_refinement.py` | generate/run nine cases, extract temperatures, score | external case-3 base project/mesh/workbook, OGS, PyVista |
| `thermal/plot_injection_production_temperature.py` | 2019–2022 injection/production temperature plot | generated thermal CSV and external workbook |
| `make_tracer_projects_portable.py` | creates path-only local-mesh copies of the 20 tracer projects | retained original projects and per-run input meshes |
| `audit_repository.py` | regenerates dependency and SHA-256 manifest CSVs | repository source tree; no simulations |

The central copies are not portable execution entry points because moving them changes their relative-path context. This is explicitly recorded in `DEPENDENCY_AUDIT.csv`; they are included as a categorized source index.
