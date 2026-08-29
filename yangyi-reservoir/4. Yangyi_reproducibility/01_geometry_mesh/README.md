# Geometry and mesh provenance

The raw inputs are ten clipped fault VTP surfaces, tabulated fault parameters, active-well collar coordinates, and a monitoring-well workbook. `generate_yangyi_dfn_split_outputs.py` reads `faults_all.csv`, constructs the 4 km cube and planar fracture network with PorePy, delegates conforming mixed-dimensional meshing to Gmsh, and converts/labels outputs for OGS. `assign_lowres_materialid.py` assigns the volumetric LRZ; `extract_six_faces_vtu.py` creates boundary meshes.

Verified generator settings: x 243300–247300 m, y 3289500–3293500 m, z 1084–5084 m; boundary/fracture/minimum sizes 200/80/20 m. `yangyi_unified_ogs.vtu` contains 35,808 points, 201,734 tetrahedra, 32,369 triangles, and 647 lines. `reservoir_with_lowres.vtu` omits the 1-D lines and is the hydraulic bulk mesh. Matrix Material 0 contains 198,046 tetrahedra, LRZ Material 12 contains 3,688 tetrahedra, and fault Materials 1–10 contain the 32,369 triangles.

Exact PorePy, Gmsh, Python, meshio, and PyVista versions were not captured. The `.msh` header records MSH format 4.1 only. Diagnostic well-intersection visualization VTUs and the duplicate DOCX tutorial were excluded.
