# Case 2 — 2D Yangyi model

## Identity

- Folder: `case2_2D_k9_k10_2e-10`
- Project: `c2_k9_k10_2e-10_200d.prj`
- OGS output prefix: `case2_k9_k10_2e-10_200d`
- k9: `2e-10 m2`
- k10: `2e-10 m2`
- k12: `2.2e-12 m2`

## Layout

- `input_mesh/`: main 2D mesh and both source meshes
- `monitoring/`: monitoring nodes and observed-data workbook
- `reference_results/`: completed log, extraction, and comparison outputs
- `output/`: created by a new OGS run
- `head_change_output/` and `comparison_output/`: created by `compare.sh`

## Run and compare

```bash
python -m pip install -r requirements.txt
chmod +x run.sh compare.sh
./run.sh
./compare.sh
```

Mesh: 16,355 nodes and 32,369 triangles. Simulation duration: 199 days with
one-day steps. Raw historical VTU files are omitted from this clean folder; a
new run recreates them under `output/`.
