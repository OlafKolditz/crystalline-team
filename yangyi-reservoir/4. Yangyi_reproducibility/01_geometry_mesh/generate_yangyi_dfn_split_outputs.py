#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Yangyi unified mixed-dimensional DFN mesh using PorePy/Gmsh.

Outputs one unified OGS-ready VTU mesh containing:
    - 3D tetrahedral rock matrix cells
    - 2D triangular fracture surface cells
    - 1D line intersection cells

MaterialID convention:
    3D tetra      -> MaterialID = 0
    2D fractures  -> MaterialID = 1, 2, 3, ...
    1D lines      -> MaterialID = 100

Required input:
    faults_all.csv

CSV columns:
    fault_id,
    center_x, center_y, center_z,
    major_axis, minor_axis,
    major_axis_angle,
    strike, dip,
    material_id

Note:
    material_id in faults_all.csv is currently mainly diagnostic.
    Final fracture MaterialIDs are assigned according to Gmsh triangle
    physical tags in ascending order.
"""

from __future__ import annotations

import os
from pathlib import Path

import meshio
import numpy as np
import pandas as pd
import porepy as pp
import pyvista as pv


# ============================================================
# User settings
# ============================================================

FAULT_CSV = Path("faults_all.csv")

OUT_DIR = Path("_out_yangyi_dfn_split")
POREPY_MESH_NAME = "yangyi_mdg"

UNIFIED_OGS_MESH_NAME = "yangyi_unified_ogs.vtu"

# Additional VTU outputs split by grid dimension
#   1D: fracture-fracture intersection line cells only
#   2D: fracture surface triangle cells only
#   3D: matrix tetrahedron cells only
#   2D+3D: fracture surfaces plus matrix, without 1D intersection lines
MESH_1D_NAME = "yangyi_1d_intersections.vtu"
MESH_2D_NAME = "yangyi_2d_fractures.vtu"
MESH_3D_NAME = "yangyi_3d_matrix.vtu"
MESH_2D3D_NAME = "yangyi_2d3d_fractures_matrix.vtu"

# Domain bounds
# 修改成你的真实模型范围
xmin, xmax = 243300.0, 247300.0
ymin, ymax = 3289500.0, 3293500.0
zmin, zmax = 1084.0, 5084.0

# Mesh size
# 根据你的模型尺度调整
mesh_size_boundary = 200.0
mesh_size_fracture = 80.0
mesh_size_min = 20.0

# 1D intersection MaterialID
LINE_MATERIAL_ID = 100


# ============================================================
# Utilities
# ============================================================

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector.")
    return v / n


def check_fault_table(faults: pd.DataFrame) -> None:
    required_cols = [
        "fault_id",
        "center_x",
        "center_y",
        "center_z",
        "major_axis",
        "minor_axis",
        "major_axis_angle",
        "strike",
        "dip",
        "material_id",
    ]

    missing = [c for c in required_cols if c not in faults.columns]
    if missing:
        raise ValueError(f"Missing columns in {FAULT_CSV}: {missing}")

    if len(faults) == 0:
        raise ValueError("Fault table is empty.")

    for _, row in faults.iterrows():
        if float(row["major_axis"]) <= 0:
            raise ValueError(f"Fault {row['fault_id']}: major_axis must be positive.")
        if float(row["minor_axis"]) <= 0:
            raise ValueError(f"Fault {row['fault_id']}: minor_axis must be positive.")


def make_elliptic_plane_fracture(
    center: np.ndarray,
    major_axis: float,
    minor_axis: float,
    major_axis_angle: float,
    strike_angle: float,
    dip_angle: float,
    num_points: int = 64,
):
    """
    Create an elliptic fracture as a polygonal PlaneFracture.

    Coordinate convention:
        x = east
        y = north
        z = upward

    strike_angle:
        clockwise from north, radians

    dip_angle:
        positive downward from horizontal, radians
    """

    center = np.asarray(center, dtype=float)

    # Strike direction in horizontal plane
    strike_vec = np.array(
        [
            np.sin(strike_angle),
            np.cos(strike_angle),
            0.0,
        ],
        dtype=float,
    )
    strike_vec = normalize(strike_vec)

    # Down-dip direction
    dip_vec = np.array(
        [
            np.cos(strike_angle) * np.cos(dip_angle),
            -np.sin(strike_angle) * np.cos(dip_angle),
            -np.sin(dip_angle),
        ],
        dtype=float,
    )
    dip_vec = normalize(dip_vec)

    # Rotate ellipse axes inside fracture plane
    major_dir = (
        np.cos(major_axis_angle) * strike_vec
        + np.sin(major_axis_angle) * dip_vec
    )
    minor_dir = (
        -np.sin(major_axis_angle) * strike_vec
        + np.cos(major_axis_angle) * dip_vec
    )

    theta = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)

    points = np.array(
        [
            center
            + major_axis * np.cos(t) * major_dir
            + minor_axis * np.sin(t) * minor_dir
            for t in theta
        ]
    ).T

    return pp.PlaneFracture(points)


def build_fractures_from_table(faults: pd.DataFrame) -> list:
    fractures = []

    for _, row in faults.iterrows():
        fault_id = row["fault_id"]

        center = np.array(
            [
                float(row["center_x"]),
                float(row["center_y"]),
                float(row["center_z"]),
            ],
            dtype=float,
        )

        major_axis = float(row["major_axis"])
        minor_axis = float(row["minor_axis"])

        if minor_axis > major_axis:
            print(
                f"Warning: Fault {fault_id} has minor_axis > major_axis. "
                "Please check whether the axes are swapped."
            )

        fracture = make_elliptic_plane_fracture(
            center=center,
            major_axis=major_axis,
            minor_axis=minor_axis,
            major_axis_angle=np.deg2rad(float(row["major_axis_angle"])),
            strike_angle=np.deg2rad(float(row["strike"])),
            dip_angle=np.deg2rad(float(row["dip"])),
            num_points=64,
        )

        fractures.append(fracture)

    return fractures


def prepare_fault_geometries(faults: pd.DataFrame) -> list[dict]:
    """
    Prepare geometric descriptors for assigning 2D triangle cells
    to fault MaterialIDs based on faults_all.csv.
    """

    fault_geometries = []

    for _, row in faults.iterrows():
        center = np.array(
            [
                float(row["center_x"]),
                float(row["center_y"]),
                float(row["center_z"]),
            ],
            dtype=float,
        )

        strike = np.deg2rad(float(row["strike"]))
        dip = np.deg2rad(float(row["dip"]))
        major_axis_angle = np.deg2rad(float(row["major_axis_angle"]))

        major_axis = float(row["major_axis"])
        minor_axis = float(row["minor_axis"])
        material_id = int(row["material_id"])

        # Strike direction
        strike_vec = np.array(
            [
                np.sin(strike),
                np.cos(strike),
                0.0,
            ],
            dtype=float,
        )
        strike_vec = normalize(strike_vec)

        # Down-dip direction
        dip_vec = np.array(
            [
                np.cos(strike) * np.cos(dip),
                -np.sin(strike) * np.cos(dip),
                -np.sin(dip),
            ],
            dtype=float,
        )
        dip_vec = normalize(dip_vec)

        # Ellipse axes in the fracture plane
        major_dir = (
            np.cos(major_axis_angle) * strike_vec
            + np.sin(major_axis_angle) * dip_vec
        )
        major_dir = normalize(major_dir)

        minor_dir = (
            -np.sin(major_axis_angle) * strike_vec
            + np.cos(major_axis_angle) * dip_vec
        )
        minor_dir = normalize(minor_dir)

        normal = normalize(np.cross(major_dir, minor_dir))

        fault_geometries.append(
            {
                "fault_id": row["fault_id"],
                "center": center,
                "major_axis": major_axis,
                "minor_axis": minor_axis,
                "major_dir": major_dir,
                "minor_dir": minor_dir,
                "normal": normal,
                "material_id": material_id,
            }
        )

    return fault_geometries


def triangle_normal(points: np.ndarray) -> np.ndarray:
    """
    Compute normal vector of one triangle.
    points shape: (3, 3)
    """
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    n = np.cross(v1, v2)

    if np.linalg.norm(n) == 0:
        return np.array([0.0, 0.0, 0.0])

    return normalize(n)


def assign_triangle_to_fault(
    tri_points: np.ndarray,
    fault_geometries: list[dict],
    distance_weight: float = 1.0,
    ellipse_weight: float = 1.0,
    normal_weight: float = 0.5,
) -> int:
    """
    Assign one triangle to the most likely fault.

    Uses:
        1. Distance from triangle center to fault plane
        2. Position inside / near the fault ellipse
        3. Triangle normal consistency with fault normal

    Returns:
        material_id of the best matching fault.
    """

    tri_center = tri_points.mean(axis=0)
    tri_n = triangle_normal(tri_points)

    best_score = np.inf
    best_material_id = None

    for fg in fault_geometries:
        center = fg["center"]
        normal = fg["normal"]
        major_dir = fg["major_dir"]
        minor_dir = fg["minor_dir"]

        rel = tri_center - center

        # Distance to fault plane
        plane_dist = abs(np.dot(rel, normal))

        # Coordinates in fault plane
        x = np.dot(rel, major_dir)
        y = np.dot(rel, minor_dir)

        # Normalized ellipse radius
        r = np.sqrt(
            (x / fg["major_axis"]) ** 2
            + (y / fg["minor_axis"]) ** 2
        )

        # Triangle normal consistency
        if np.linalg.norm(tri_n) > 0:
            normal_misfit = 1.0 - abs(np.dot(tri_n, normal))
        else:
            normal_misfit = 1.0

        # Score: smaller is better
        score = (
            distance_weight * plane_dist
            + ellipse_weight * r
            + normal_weight * normal_misfit
        )

        if score < best_score:
            best_score = score
            best_material_id = fg["material_id"]

    if best_material_id is None:
        raise RuntimeError("Could not assign triangle to any fault.")

    return int(best_material_id)



# ============================================================
# VTU subset export helpers
# ============================================================

def save_cell_subset_by_mask(
    grid: pv.UnstructuredGrid,
    mask: np.ndarray,
    out_file: Path,
    label: str,
) -> pv.UnstructuredGrid | None:
    """
    Save a subset of cells from a PyVista UnstructuredGrid.

    Parameters
    ----------
    grid
        Full mixed-dimensional grid.
    mask
        Boolean mask with length equal to grid.n_cells.
    out_file
        Output VTU file path.
    label
        Human-readable name printed in the log.

    Returns
    -------
    subset or None
        The extracted subset grid, or None if no cell satisfies the mask.
    """

    mask = np.asarray(mask, dtype=bool)

    if mask.size != grid.n_cells:
        raise ValueError(
            f"Mask size {mask.size} does not match number of cells {grid.n_cells}."
        )

    n_cells = int(np.sum(mask))
    if n_cells == 0:
        print(f"Warning: {label} subset is empty. Skip saving: {out_file}")
        return None

    subset = grid.extract_cells(mask)
    subset.save(out_file, binary=False)

    print(f"Saved {label}: {out_file}")
    print(f"  cells: {subset.n_cells}, points: {subset.n_points}")

    if "MaterialIDs" in subset.cell_data:
        mids = np.asarray(subset.cell_data["MaterialIDs"], dtype=int)
        print("  MaterialIDs:")
        for mid in sorted(set(mids)):
            print(f"    {mid}: {np.sum(mids == mid)} cells")

    return subset


def save_dimension_split_outputs(
    grid: pv.UnstructuredGrid,
    out_dir: Path,
) -> dict[str, Path]:
    """
    Save 1D, 2D, 3D, and combined 2D+3D VTU files from the unified grid.

    The function uses the cell_data array 'grid_dim':
        grid_dim = 1 -> 1D line cells
        grid_dim = 2 -> 2D triangle cells
        grid_dim = 3 -> 3D tetrahedron cells
    """

    if "grid_dim" not in grid.cell_data:
        raise RuntimeError("Cannot split mesh: cell_data['grid_dim'] is missing.")

    out_dir.mkdir(parents=True, exist_ok=True)
    gd = np.asarray(grid.cell_data["grid_dim"], dtype=int)

    outputs = {
        "1d": out_dir / MESH_1D_NAME,
        "2d": out_dir / MESH_2D_NAME,
        "3d": out_dir / MESH_3D_NAME,
        "2d3d": out_dir / MESH_2D3D_NAME,
    }

    print("\nSaving dimension-split VTU files...")
    save_cell_subset_by_mask(grid, gd == 1, outputs["1d"], "1D intersection mesh")
    save_cell_subset_by_mask(grid, gd == 2, outputs["2d"], "2D fracture mesh")
    save_cell_subset_by_mask(grid, gd == 3, outputs["3d"], "3D matrix mesh")
    save_cell_subset_by_mask(
        grid,
        np.isin(gd, [2, 3]),
        outputs["2d3d"],
        "combined 2D+3D fracture-matrix mesh",
    )

    return outputs

# ============================================================
# Convert Gmsh .msh to unified OGS VTU
# ============================================================

def convert_msh_to_unified_ogs_vtu(
    msh_file: Path,
    out_file: Path,
    faults: pd.DataFrame,
    line_material_id: int = 100,
    tetra_matrix_tag: int | None = None,
) -> None:
    """
    Convert Gmsh .msh to one unified OGS-ready VTU.

    Final MaterialIDs:
        3D tetra      -> 0
        2D triangle   -> 1, 2, 3, ... according to ascending triangle physical tags
        1D line       -> line_material_id
    """

    if not msh_file.exists():
        raise FileNotFoundError(f"Cannot find msh file: {msh_file}")

    print(f"\nReading Gmsh mesh: {msh_file}")
    msh = meshio.read(msh_file)

    fault_geometries = prepare_fault_geometries(faults)

    gmsh_phys = msh.cell_data_dict.get("gmsh:physical", {})

    if not gmsh_phys:
        raise RuntimeError("No gmsh:physical tags found in msh file.")

    # ------------------------------------------------------------
    # Collect physical tags by cell type
    # ------------------------------------------------------------
    line_tags: set[int] = set()
    triangle_tags: set[int] = set()
    tetra_tags: set[int] = set()

    for block in msh.cells:
        ctype = block.type

        if ctype not in gmsh_phys:
            continue

        tags = np.asarray(gmsh_phys[ctype], dtype=int)

        if ctype == "line":
            line_tags.update(tags.tolist())
        elif ctype == "triangle":
            triangle_tags.update(tags.tolist())
        elif ctype == "tetra":
            tetra_tags.update(tags.tolist())

    line_tags = set(sorted(line_tags))
    triangle_tags_sorted = sorted(triangle_tags)
    tetra_tags_sorted = sorted(tetra_tags)

    print("\nPhysical tags found in msh:")
    print("  line tags:    ", sorted(line_tags))
    print("  triangle tags:", triangle_tags_sorted)
    print("  tetra tags:   ", tetra_tags_sorted)

    if not triangle_tags_sorted:
        raise RuntimeError("No triangle physical tags found. No 2D fractures detected.")

    if len(tetra_tags_sorted) != 1 and tetra_matrix_tag is None:
        raise RuntimeError(
            f"Expected exactly one tetra physical tag for matrix, found {tetra_tags_sorted}. "
            "Please specify tetra_matrix_tag manually."
        )

    if tetra_matrix_tag is None:
        tetra_matrix_tag = tetra_tags_sorted[0]

    # Map triangle physical tags to MaterialIDs 1..N
    triangle_tag_to_material_id = {
        tag: i + 1 for i, tag in enumerate(triangle_tags_sorted)
    }

    print("\nTriangle physical tag -> OGS MaterialID:")
    for tag, mid in triangle_tag_to_material_id.items():
        print(f"  gmsh tag {tag} -> MaterialID {mid}")

    print(f"\nTetra physical tag {tetra_matrix_tag} -> MaterialID 0")
    print(f"All line physical tags -> MaterialID {line_material_id}")

    vtk_cells: list[int] = []
    vtk_types: list[int] = []
    material_ids: list[int] = []
    grid_dim: list[int] = []
    gmsh_physical_tags: list[int] = []

    # ------------------------------------------------------------
    # Build unified VTU cell array
    # ------------------------------------------------------------
    for block in msh.cells:
        ctype = block.type
        data = block.data

        if ctype not in gmsh_phys:
            print(f"Skipping {ctype}: no gmsh physical tags")
            continue

        phys = np.asarray(gmsh_phys[ctype], dtype=int)

        if ctype == "vertex":
            # Vertex physical groups are not needed for current flow/tracer setup.
            continue

        if ctype == "line":
            for conn, tag in zip(data, phys):
                vtk_cells.extend([2, int(conn[0]), int(conn[1])])
                vtk_types.append(3)  # VTK_LINE

                material_ids.append(line_material_id)
                grid_dim.append(1)
                gmsh_physical_tags.append(int(tag))

        elif ctype == "triangle":
            for conn, tag in zip(data, phys):
                tag = int(tag)

                p0 = msh.points[int(conn[0])]
                p1 = msh.points[int(conn[1])]
                p2 = msh.points[int(conn[2])]

                tri_points = np.array([p0, p1, p2], dtype=float)

                assigned_mid = assign_triangle_to_fault(
                    tri_points=tri_points,
                    fault_geometries=fault_geometries,
                )

                vtk_cells.extend([3, int(conn[0]), int(conn[1]), int(conn[2])])
                vtk_types.append(5)  # VTK_TRIANGLE

                material_ids.append(assigned_mid)
                grid_dim.append(2)
                gmsh_physical_tags.append(tag)

        elif ctype == "tetra":
            for conn, tag in zip(data, phys):
                tag = int(tag)

                if tag != tetra_matrix_tag:
                    raise RuntimeError(
                        f"Unexpected tetra physical tag {tag}; expected {tetra_matrix_tag}."
                    )

                vtk_cells.extend(
                    [
                        4,
                        int(conn[0]),
                        int(conn[1]),
                        int(conn[2]),
                        int(conn[3]),
                    ]
                )
                vtk_types.append(10)  # VTK_TETRA

                material_ids.append(0)
                grid_dim.append(3)
                gmsh_physical_tags.append(tag)

        else:
            print(f"Skipping unsupported cell type: {ctype}")

    grid = pv.UnstructuredGrid(
        np.asarray(vtk_cells, dtype=np.int64),
        np.asarray(vtk_types, dtype=np.uint8),
        msh.points,
    )

    grid.cell_data["MaterialIDs"] = np.asarray(material_ids, dtype=np.int32)
    grid.cell_data["grid_dim"] = np.asarray(grid_dim, dtype=np.int32)
    grid.cell_data["gmsh_physical_tag"] = np.asarray(gmsh_physical_tags, dtype=np.int32)
    grid.cell_data["cell_id"] = np.arange(grid.n_cells, dtype=np.int32)

    grid.save(out_file, binary=False)

    # Also save separated dimensional meshes and the combined 2D+3D mesh.
    split_outputs = save_dimension_split_outputs(grid, out_file.parent)

    print(f"\nSaved unified OGS mesh: {out_file}")
    print(grid)
    print("Cell types:", sorted(set(grid.celltypes)))

    mids = np.asarray(grid.cell_data["MaterialIDs"], dtype=int)
    gd = np.asarray(grid.cell_data["grid_dim"], dtype=int)

    print("\nMaterialID summary:")
    for mid in sorted(set(mids)):
        print(f"  MaterialID {mid}: {np.sum(mids == mid)} cells")

    print("\nGrid dimension summary:")
    for d in sorted(set(gd)):
        print(f"  grid_dim {d}: {np.sum(gd == d)} cells")


# ============================================================
# Mesh quality / topology checks
# ============================================================

def check_unified_mesh(mesh_file: Path) -> None:
    """
    Check whether:
        - 2D triangles are tetra faces
        - 1D lines are triangle edges
    """

    from collections import Counter

    print(f"\nChecking unified mesh topology: {mesh_file}")
    mesh = pv.read(mesh_file)

    celltypes = np.asarray(mesh.celltypes)
    cells = np.asarray(mesh.cells)

    print(mesh)
    print("Cell types:", sorted(set(celltypes)))

    print("\nCell type counts:")
    for ct in sorted(set(celltypes)):
        print(f"  type {ct}: {np.sum(celltypes == ct)} cells")

    tet_faces = Counter()
    tri_faces = []
    line_edges = []

    offset = 0

    for ct in celltypes:
        n = int(cells[offset])
        ids = tuple(int(x) for x in cells[offset + 1: offset + 1 + n])
        offset += n + 1

        if ct == 10:
            a, b, c, d = ids
            for f in [
                tuple(sorted((a, b, c))),
                tuple(sorted((a, b, d))),
                tuple(sorted((a, c, d))),
                tuple(sorted((b, c, d))),
            ]:
                tet_faces[f] += 1

        elif ct == 5:
            tri_faces.append(tuple(sorted(ids)))

        elif ct == 3:
            line_edges.append(tuple(sorted(ids)))

    # Triangle vs tetra face
    matches = 0
    internal = 0
    boundary = 0
    missing = 0

    for f in tri_faces:
        count = tet_faces.get(f, 0)

        if count > 0:
            matches += 1
            if count == 1:
                boundary += 1
            elif count == 2:
                internal += 1
        else:
            missing += 1

    print("\nTriangle vs tetra-face check:")
    print("  triangle cells:", len(tri_faces))
    print("  matching tetra faces:", matches, "/", len(tri_faces))
    print("  internal matches:", internal)
    print("  boundary matches:", boundary)
    print("  missing:", missing)

    # Line vs triangle edge
    tri_edges = Counter()

    for a, b, c in tri_faces:
        for e in [
            tuple(sorted((a, b))),
            tuple(sorted((a, c))),
            tuple(sorted((b, c))),
        ]:
            tri_edges[e] += 1

    line_matches = 0
    line_missing = 0

    for e in line_edges:
        if tri_edges.get(e, 0) > 0:
            line_matches += 1
        else:
            line_missing += 1

    print("\nLine vs triangle-edge check:")
    print("  line cells:", len(line_edges))
    print("  matching triangle edges:", line_matches, "/", len(line_edges))
    print("  missing:", line_missing)


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    original_dir = Path.cwd()

    if not FAULT_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find {FAULT_CSV}. Put faults_all.csv in the same folder as this script."
        )

    faults = pd.read_csv(FAULT_CSV)
    check_fault_table(faults)

    print("Read fault table:")
    print(
        faults[
            [
                "fault_id",
                "center_x",
                "center_y",
                "center_z",
                "major_axis",
                "minor_axis",
                "strike",
                "dip",
                "material_id",
            ]
        ]
    )
    print(f"\nNumber of fractures in table: {len(faults)}")

    fractures = build_fractures_from_table(faults)

    bounding_box = {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "zmin": zmin,
        "zmax": zmax,
    }

    domain = pp.Domain(bounding_box=bounding_box)

    print("\nCreating fracture network...")
    network = pp.create_fracture_network(
        fractures=fractures,
        domain=domain,
    )

    mesh_args = {
        "cell_size_boundary": mesh_size_boundary,
        "cell_size_fracture": mesh_size_fracture,
        "cell_size_min": mesh_size_min,
        "export": True,
        "filename": POREPY_MESH_NAME,
    }

    print("\nCreating mixed-dimensional grid...")
    print("Mesh arguments:", mesh_args)

    os.chdir(OUT_DIR)
    try:
        _ = pp.create_mdg("simplex", mesh_args, network)
    finally:
        os.chdir(original_dir)

    # ------------------------------------------------------------
    # Find generated .msh file
    # ------------------------------------------------------------
    print("\nSearching for generated Gmsh .msh file...")

    msh_files = sorted(OUT_DIR.glob("*.msh"))

    if not msh_files:
        raise FileNotFoundError(
            f"No .msh file found in {OUT_DIR}. "
            "Check whether mesh_args['export'] = True and PorePy/Gmsh exported a .msh file."
        )

    print("Found msh files:")
    for f in msh_files:
        print(f"  {f}")

    # Use newest .msh if multiple exist
    msh_file = max(msh_files, key=lambda p: p.stat().st_mtime)

    unified_ogs_file = OUT_DIR / UNIFIED_OGS_MESH_NAME

    convert_msh_to_unified_ogs_vtu(
        msh_file=msh_file,
        out_file=unified_ogs_file,
        faults=faults,
        line_material_id=LINE_MATERIAL_ID,
        tetra_matrix_tag=None,
    )

    check_unified_mesh(unified_ogs_file)

    print("\nDone.")
    print("Main outputs:")
    print(f"  Unified OGS mixed-dimensional mesh: {unified_ogs_file}")
    print(f"  1D intersection mesh:             {OUT_DIR / MESH_1D_NAME}")
    print(f"  2D fracture mesh:                 {OUT_DIR / MESH_2D_NAME}")
    print(f"  3D matrix mesh:                   {OUT_DIR / MESH_3D_NAME}")
    print(f"  2D+3D fracture-matrix mesh:       {OUT_DIR / MESH_2D3D_NAME}")


if __name__ == "__main__":
    main()
