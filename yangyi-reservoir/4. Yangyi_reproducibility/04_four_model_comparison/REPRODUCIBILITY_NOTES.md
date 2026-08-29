# Four-model structural comparison

Mapping: `DFNM` = DFNM-noLRZ (matrix + faults); `DFNM-LRZ` = Full DFNM (matrix + faults + LRZ); `DFN` = Surface DFN (faults only); `DFN-LRZ` = faults + LRZ without matrix. The first, second, and fourth use ZK403/ZK208/ZK203. Surface DFN uses only balanced ZK403/ZK208 forcing, so its comparison with DFN-LRZ is not a strict LRZ-only ablation.

`permeability_transfer.csv` and the projects verify that `joint_007` values were transferred without independent recalibration. Full configurations use 200 daily steps; Surface DFN retains 199 daily steps. The field script slices 3-D matrix cells at the mesh centre with `normal='y'`; this is a constant-y X-Z plane, not a Z-Y plane through ZK203 depth.

Run the four final projects using `run_all.sh`; then use `plot_well_pressure_comparisons_portable.py`, `plot_monitoring_head_comparisons_portable.py --model all`, and the original `plot_final_field_comparisons.py`. The two portable copies change only input paths to retained source/mapping files; their original scripts are preserved. `build_cases.py` is provenance for generation and requires external original base directories, so it is not needed to rerun the already generated final projects.
