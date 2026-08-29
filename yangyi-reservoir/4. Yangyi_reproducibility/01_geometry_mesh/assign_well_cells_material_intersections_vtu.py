#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find intersections between three well trajectories and cells with MaterialIDs 1-12
in reservoir_with_lowres.vtu, and write the results as VTU files.

This script is modified from assign_well_cells.py.

Default inputs, relative to this script:
    ./well/YY_Well_ZK203_Planning_collar.csv
    ./well/YY_Well_ZK208_Planning_collar.csv
    ./well/YY_Well_ZK403_Planning_collar.csv
    ./_out_yangyi_dfn/reservoir_with_lowres.vtu

Default outputs:
    ./_out_yangyi_dfn/well_material_intersections/
        all_wells_MID1-12_intersection_cells.vtu
        all_wells_MID1-12_intersection_points.vtu
        ZK203_MID1-12_intersection_cells.vtu
        ZK203_MID1-12_intersection_points.vtu
        ZK208_MID1-12_intersection_cells.vtu
        ZK208_MID1-12_intersection_points.vtu
        ZK403_MID1-12_intersection_cells.vtu
        ZK403_MID1-12_intersection_points.vtu
        well_materialID_1_12_intersections.csv

Notes
-----
1. If the well CSV contains only collar coordinates, the well is treated as a
   vertical line from collar elevation down to the bottom of the mesh.
2. If a well trajectory CSV is available, set USE_TRAJECTORY_IF_AVAILABLE=True
   and prepare files named, for example:
       ./well/ZK203_trajectory.csv
   with columns:
       x,y,z
   or:
       POINT_X,POINT_Y,RASTERVALU
3. MaterialID array name is automatically detected from common names.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyvista as pv


# ============================================================
# User settings
# ============================================================

WELL_NAMES = ["ZK203", "ZK208", "ZK403"]

RESERVOIR_MESH_NAME = "reservoir_with_lowres.vtu"
RESERVOIR_MESH_DIR = "_out_yangyi_dfn_split"

WELL_DIR_NAME = "well"
OUT_SUBDIR_NAME = "well_material_intersections"

TARGET_MATERIAL_IDS = list(range(1, 13))

# If True, the script first tries to read ./well/ZK203_trajectory.csv etc.
# If not found, it falls back to the original Planning_collar CSV.
USE_TRAJECTORY_IF_AVAILABLE = True

# Extra length below mesh bottom, to make sure the vertical well line crosses
# the whole model. Unit is the same as the mesh coordinates, normally meter.
EXTRA_DEPTH_BELOW_MESH = 50.0

# Tolerance for find_cells_along_line. Increase to 1.0, 2.0 or 5.0 if strict
# line-cell intersection misses very close cells.
LINE_TOLERANCE = 1.0e-6

# If one well intersects many cells, keep all of them. If you only want one
# cell per MaterialID, set this to True.
KEEP_ONE_CELL_PER_MATERIAL_ID = False


# ============================================================
# Basic readers
# ============================================================

def detect_material_array(mesh: pv.DataSet) -> str:
    """Return the most likely MaterialID cell-data array name."""
    candidates = ["MaterialIDs", "MaterialID", "material_id", "material_ids"]
    for name in candidates:
        if name in mesh.cell_data:
            return name
    available = list(mesh.cell_data.keys())
    raise KeyError(
        "Cannot find a MaterialID cell-data array. "
        f"Tried {candidates}. Available cell arrays: {available}"
    )


def get_column(row: dict, names: Iterable[str]) -> float:
    """Read a float from a CSV row using one of several possible column names."""
    for name in names:
        if name in row and row[name] not in (None, ""):
            return float(row[name])
    raise KeyError(f"Cannot find any of columns {list(names)} in CSV row: {row}")


def read_collar_point(well_dir: Path, name: str) -> tuple[np.ndarray, Path]:
    """Read one well collar point from the original Planning_collar CSV."""
    csv_path = well_dir / f"YY_Well_{name}_Planning_collar.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing well collar file: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        row = next(reader)

    x = get_column(row, ["POINT_X", "x", "X"])
    y = get_column(row, ["POINT_Y", "y", "Y"])
    z = get_column(row, ["RASTERVALU", "POINT_Z", "z", "Z", "elevation"])

    return np.array([x, y, z], dtype=float), csv_path


def read_trajectory_points(well_dir: Path, name: str) -> tuple[np.ndarray | None, Path | None]:
    """Read optional well trajectory points."""
    possible = [
        well_dir / f"{name}_trajectory.csv",
        well_dir / f"YY_Well_{name}_trajectory.csv",
        well_dir / f"YY_Well_{name}_Planning_trajectory.csv",
    ]

    for csv_path in possible:
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}

        def find_col(options: list[str]) -> str:
            for opt in options:
                if opt.lower() in cols:
                    return cols[opt.lower()]
            raise KeyError(f"Cannot find columns {options} in {csv_path}")

        xcol = find_col(["x", "POINT_X", "Easting"])
        ycol = find_col(["y", "POINT_Y", "Northing"])
        zcol = find_col(["z", "POINT_Z", "RASTERVALU", "Elevation"])

        pts = df[[xcol, ycol, zcol]].to_numpy(dtype=float)
        if pts.shape[0] < 2:
            raise ValueError(f"Trajectory file has fewer than 2 points: {csv_path}")
        return pts, csv_path

    return None, None


def read_well_lines(well_dir: Path, mesh: pv.DataSet) -> list[dict]:
    """
    Read three wells.

    If trajectory files exist, use them. Otherwise create vertical well lines
    from collar z down to mesh bottom.
    """
    zmin = float(mesh.bounds[4])
    wells = []

    for name in WELL_NAMES:
        trajectory = None
        source_file = None

        if USE_TRAJECTORY_IF_AVAILABLE:
            trajectory, source_file = read_trajectory_points(well_dir, name)

        if trajectory is None:
            collar, source_file = read_collar_point(well_dir, name)
            bottom = np.array([collar[0], collar[1], zmin - EXTRA_DEPTH_BELOW_MESH], dtype=float)
            trajectory = np.vstack([collar, bottom])

        wells.append(
            {
                "name": name,
                "well_id": int(name.replace("ZK", "")),
                "points": trajectory,
                "file": str(source_file),
            }
        )

    return wells


# ============================================================
# Geometry and extraction
# ============================================================

def cells_along_polyline(mesh: pv.DataSet, points: np.ndarray, tolerance: float) -> np.ndarray:
    """Return unique cell ids intersected by a polyline."""
    all_ids: list[int] = []

    for p0, p1 in zip(points[:-1], points[1:]):
        # Skip zero-length segments
        if np.linalg.norm(p1 - p0) == 0:
            continue

        ids = mesh.find_cells_along_line(p0, p1, tolerance=tolerance)
        if ids is None:
            continue
        all_ids.extend([int(i) for i in np.asarray(ids).ravel()])

    if not all_ids:
        return np.array([], dtype=np.int64)

    return np.unique(np.asarray(all_ids, dtype=np.int64))


def make_point_vtu(points: np.ndarray, point_data: dict[str, np.ndarray]) -> pv.UnstructuredGrid:
    """Create a VTU containing VTK_VERTEX cells at the supplied points."""
    points = np.asarray(points, dtype=float)

    if points.size == 0:
        # Empty VTU with no cells.
        return pv.UnstructuredGrid()

    n = points.shape[0]
    cells = np.empty((n, 2), dtype=np.int64)
    cells[:, 0] = 1
    cells[:, 1] = np.arange(n, dtype=np.int64)

    celltypes = np.full(n, pv.CellType.VERTEX, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells.ravel(), celltypes, points)

    for key, values in point_data.items():
        arr = np.asarray(values)
        if arr.shape[0] == n:
            # Store on both point_data and cell_data for convenience in ParaView.
            grid.point_data[key] = arr
            grid.cell_data[key] = arr.copy()

    return grid


def append_grids(grids: list[pv.DataSet]) -> pv.UnstructuredGrid:
    """Append non-empty grids into one UnstructuredGrid."""
    non_empty = [g for g in grids if g is not None and g.n_cells > 0]
    if not non_empty:
        return pv.UnstructuredGrid()
    return non_empty[0].merge(non_empty[1:]) if len(non_empty) > 1 else non_empty[0]


def extract_intersections() -> None:
    root = Path(__file__).resolve().parent
    reservoir_mesh_file = root / RESERVOIR_MESH_DIR / RESERVOIR_MESH_NAME
    well_dir = root / WELL_DIR_NAME
    out_dir = root / RESERVOIR_MESH_DIR / OUT_SUBDIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    print("读取 reservoir 网格:", reservoir_mesh_file)
    if not reservoir_mesh_file.exists():
        raise FileNotFoundError(f"Cannot find reservoir mesh: {reservoir_mesh_file}")
    reservoir = pv.read(str(reservoir_mesh_file))

    material_array = detect_material_array(reservoir)
    material_ids = np.asarray(reservoir.cell_data[material_array], dtype=int)
    target_material_ids = np.asarray(TARGET_MATERIAL_IDS, dtype=int)

    print(f"使用 MaterialID 数组: {material_array}")
    print(f"目标 MaterialIDs: {TARGET_MATERIAL_IDS}")

    target_mask = np.isin(material_ids, target_material_ids)
    target_cell_ids_global = np.where(target_mask)[0]
    if target_cell_ids_global.size == 0:
        raise RuntimeError(
            f"No cells found with {material_array} in {TARGET_MATERIAL_IDS}. "
            f"Available IDs: {sorted(set(material_ids.tolist()))}"
        )

    print(f"目标 MaterialID 单元数: {target_cell_ids_global.size}")
    target_mesh = reservoir.extract_cells(target_cell_ids_global)
    target_mesh.cell_data["original_cell_id"] = target_cell_ids_global.astype(np.int64)

    print("读取井信息目录:", well_dir)
    wells = read_well_lines(well_dir, reservoir)

    all_cell_outputs: list[pv.UnstructuredGrid] = []
    all_point_outputs: list[pv.UnstructuredGrid] = []
    summary_rows: list[dict] = []

    for well in wells:
        name = well["name"]
        well_id = int(well["well_id"])
        pts = np.asarray(well["points"], dtype=float)

        print(f"\n处理 {name}: {well['file']}")
        print(f"  轨迹点数: {pts.shape[0]}")

        local_ids = cells_along_polyline(target_mesh, pts, LINE_TOLERANCE)

        if local_ids.size == 0:
            print(f"  警告: {name} 没有与 MaterialID 1-12 单元相交。")
            continue

        original_ids = np.asarray(target_mesh.cell_data["original_cell_id"])[local_ids].astype(np.int64)
        mids = material_ids[original_ids]

        if KEEP_ONE_CELL_PER_MATERIAL_ID:
            keep_local = []
            used_mid = set()
            for lid, mid in zip(local_ids, mids):
                if int(mid) in used_mid:
                    continue
                keep_local.append(int(lid))
                used_mid.add(int(mid))
            local_ids = np.asarray(keep_local, dtype=np.int64)
            original_ids = np.asarray(target_mesh.cell_data["original_cell_id"])[local_ids].astype(np.int64)
            mids = material_ids[original_ids]

        well_cells = target_mesh.extract_cells(local_ids)
        well_cells.cell_data["WellID"] = np.full(well_cells.n_cells, well_id, dtype=np.int32)
        well_cells.cell_data["WellNameCode"] = np.full(well_cells.n_cells, well_id, dtype=np.int32)
        well_cells.cell_data["intersection_rank"] = np.arange(well_cells.n_cells, dtype=np.int32)

        # Use cell centers as representative intersection points. For grid-based
        # monitoring/extraction this is usually more useful than exact geometric
        # segment-triangle crossing points, because it points to the selected cell.
        centers = well_cells.cell_centers().points
        point_grid = make_point_vtu(
            centers,
            {
                "WellID": np.full(centers.shape[0], well_id, dtype=np.int32),
                "MaterialIDs": np.asarray(well_cells.cell_data[material_array], dtype=np.int32),
                "original_cell_id": np.asarray(well_cells.cell_data["original_cell_id"], dtype=np.int64),
                "intersection_rank": np.arange(centers.shape[0], dtype=np.int32),
            },
        )

        cell_out = out_dir / f"{name}_MID1-12_intersection_cells.vtu"
        point_out = out_dir / f"{name}_MID1-12_intersection_points.vtu"
        well_cells.save(str(cell_out))
        point_grid.save(str(point_out))

        print(f"  相交单元数: {well_cells.n_cells}")
        print(f"  MaterialIDs: {sorted(set(int(v) for v in mids.tolist()))}")
        print(f"  保存: {cell_out}")
        print(f"  保存: {point_out}")

        all_cell_outputs.append(well_cells)
        all_point_outputs.append(point_grid)

        for rank, (orig_id, mid, center) in enumerate(zip(original_ids, mids, centers)):
            summary_rows.append(
                {
                    "well": name,
                    "well_id": well_id,
                    "rank": rank,
                    "original_cell_id": int(orig_id),
                    "material_id": int(mid),
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "z": float(center[2]),
                    "source_file": well["file"],
                }
            )

    all_cells = append_grids(all_cell_outputs)
    all_points = append_grids(all_point_outputs)

    all_cells_out = out_dir / "all_wells_MID1-12_intersection_cells.vtu"
    all_points_out = out_dir / "all_wells_MID1-12_intersection_points.vtu"
    summary_csv = out_dir / "well_materialID_1_12_intersections.csv"

    if all_cells.n_cells > 0:
        all_cells.save(str(all_cells_out))
        print("\n保存全部井相交单元 VTU:", all_cells_out)
    else:
        print("\n没有任何相交单元，未保存 all_wells cell VTU。")

    if all_points.n_cells > 0:
        all_points.save(str(all_points_out))
        print("保存全部井相交点 VTU:", all_points_out)
    else:
        print("没有任何相交点，未保存 all_wells point VTU。")

    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print("保存交点/单元汇总 CSV:", summary_csv)
    print("\n完成。")


if __name__ == "__main__":
    extract_intersections()
