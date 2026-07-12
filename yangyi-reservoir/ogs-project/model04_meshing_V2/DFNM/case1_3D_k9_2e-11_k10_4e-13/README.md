# Yangyi case c1_k9_2e-11_k10_4e-13_200d

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
- Validation status: project initialization and first-step startup succeeded
- Expected runtime: about 40 minutes on the machine used for the original c14 run
- Key permeability values:
  - Material 9: 2e-11 m2
  - Material 10: 4e-13 m2
  - Material 12: 2.5e-12 m2
- Output prefix: `case1_k9_2e-11_k10_4e-13_200d`

## Folder contents

- `c1_k9_2e-11_k10_4e-13_200d.prj`: portable OGS project file
- `input_mesh/`: bulk mesh and boundary meshes
- `source_terms/`: ZK403, ZK208, and ZK203 source/sink node meshes
- `monitoring/`: monitoring-node mesh and well-coordinate table
- `monitoring/yangyi_after_20190113_for_ogs.xlsx`: observed monitoring dataset
- `reference_results/`: results retained from the original c14 package for
  reference only; they do not represent the new case1 permeabilities
- `run.sh`: Linux/macOS run helper
- `compare_simulated_observed_head_change.py`: reusable comparison code
- `compare.sh`: Linux/macOS comparison helper
- `requirements.txt`: Python packages required by the comparison code

The full 201-file VTU/PVD series is not included. Running this project creates
the full case1 VTU/PVD output series. Files under `reference_results/` belong to
the original c14 case and must not be interpreted as case1 results.

## Run on Linux or macOS

Ensure `ogs` is available on `PATH`, enter this folder, and run:

```bash
chmod +x run.sh
./run.sh
```

Or run OGS directly:

```bash
mkdir -p output
ogs c1_k9_2e-11_k10_4e-13_200d.prj -o output
```

## Run on Windows PowerShell

Enter the extracted folder and run:

```powershell
New-Item -ItemType Directory -Force output
ogs.exe c1_k9_2e-11_k10_4e-13_200d.prj -o output
```

## Notes

OGS reports a material-definition-count warning and a fixed-output-time warning
during initialization. The same warnings were present in the successful
original c14 run and did not stop initialization of this case1 project.

## Rerun the simulated-versus-observed comparison

After the OGS run has created
`output/case1_k9_2e-11_k10_4e-13_200d.pvd` and its VTU files,
install the Python dependencies and run:

```bash
python -m pip install -r requirements.txt
chmod +x compare.sh
./compare.sh
```

The comparison results will be written to `comparison_output/`. The equivalent
direct command is:

```bash
python compare_simulated_observed_head_change.py --pvd output/case1_k9_2e-11_k10_4e-13_200d.pvd --monitor_csv monitoring/recommended_monitoring_points_all7_ZK501_bedrock.csv --obs_xlsx monitoring/yangyi_after_20190113_for_ogs.xlsx --obs_sheet Monitor_compare_after20190113 --out_dir comparison_case1_k9_2e-11_k10_4e-13_200d
```
