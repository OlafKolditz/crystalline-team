# Yangyi case c2_k9_k10_2e-10_200d

This is a self-contained OpenGeoSys 6 liquid-flow case covering 200 days after
2019-01-13. All files required to rerun the model are included. Paths in the
shared project file are relative to this folder, so the package can be moved or
extracted anywhere.

## Model configuration

- OGS version used: 6.5.8
- Process: `LIQUID_FLOW`
- Bulk mesh: 35,808 nodes and 234,103 elements
  - 201,734 tetrahedra
  - 32,369 triangles
- Duration: 17,280,000 seconds (200 days)
- Time step: 86,400 seconds (1 day)
- Completed reference run: 200 accepted steps, 0 rejected steps
- Reference computation time: 2,393.02 seconds (about 39 min 53 s)
- Key permeability values:
  - Material 9: 2e-10 m2
  - Material 10: 2e-10 m2
  - Material 12: 2.5e-12 m2
- Output prefix: `case2_k9_k10_2e-10_200d`

## Folder contents

- `c2_k9_k10_2e-10_200d.prj`: portable OGS project file
- `input_mesh/`: bulk mesh and boundary meshes
- `source_terms/`: ZK403, ZK208, and ZK203 source/sink node meshes
- `monitoring/`: monitoring-node mesh and well-coordinate table
- `monitoring/yangyi_after_20190113_for_ogs.xlsx`: observed monitoring dataset
- `reference_results/`: completed log, extracted well time series, error tables,
  and observed-versus-simulated plots
- `run.sh`: Linux/macOS run helper
- `compare_simulated_observed_head_change.py`: reusable comparison code
- `compare.sh`: Linux/macOS comparison helper
- `requirements.txt`: Python packages required by the comparison code

The full 201-file VTU/PVD series is not duplicated in this compact sharing
package because it is about 1 GB. The included reference results contain the
well-level simulation data and plots. Rerunning the project recreates the full
VTU/PVD output series.

## Run on Linux or macOS

Ensure `ogs` is available on `PATH`, enter this folder, and run:

```bash
chmod +x run.sh
./run.sh
```

Or run OGS directly:

```bash
mkdir -p output
ogs c2_k9_k10_2e-10_200d.prj -o output
```

## Run on Windows PowerShell

Enter the extracted folder and run:

```powershell
New-Item -ItemType Directory -Force output
ogs.exe c2_k9_k10_2e-10_200d.prj -o output
```

## Notes

OGS reports a material-definition-count warning and a fixed-output-time warning
during initialization. These warnings were present in the successful reference
run; the solver completed all 200 steps without rejection.

## Rerun the simulated-versus-observed comparison

After the OGS run has created `output/case2_k9_k10_2e-10_200d.pvd` and its VTU files,
install the Python dependencies and run:

```bash
python -m pip install -r requirements.txt
chmod +x compare.sh
./compare.sh
```

The comparison results will be written to `comparison_output/`. The equivalent
direct command is:

```bash
python compare_simulated_observed_head_change.py --pvd output/case2_k9_k10_2e-10_200d.pvd --monitor_csv monitoring/recommended_monitoring_points_all7_ZK501_bedrock.csv --obs_xlsx monitoring/yangyi_after_20190113_for_ogs.xlsx --obs_sheet Monitor_compare_after20190113 --out_dir comparison_case2_k9_k10_2e-10_200d
```
