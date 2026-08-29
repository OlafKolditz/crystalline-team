#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract six boundary faces from a VTU mesh and save each face as VTU.

Input by default:
    reservoir_with_lowres.vtu

Outputs by default:
    six_faces_vtu/yangyi_face_x_min.vtu
    six_faces_vtu/yangyi_face_x_max.vtu
    six_faces_vtu/yangyi_face_y_min.vtu
    six_faces_vtu/yangyi_face_y_max.vtu
    six_faces_vtu/yangyi_face_z_min.vtu
    six_faces_vtu/yangyi_face_z_max.vtu

Compared with the original script:
    1. Output format is .vtu instead of .vtp.
    2. Boundary meshes are saved as vtkUnstructuredGrid.
    3. Original point ids are preserved when VTK supports it.
    4. A UInt64 point-data array named bulk_node_ids is written for OGS-style
       boundary meshes, using original bulk point ids when available.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import vtk


FACE_NAMES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


def read_unstructured_grid(path: str | Path) -> vtk.vtkUnstructuredGrid:
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    ug = reader.GetOutput()
    if ug is None or ug.GetNumberOfCells() == 0:
        raise RuntimeError(f"Failed to read a non-empty unstructured grid: {path}")
    return ug


def extract_boundary_with_original_ids(ug: vtk.vtkUnstructuredGrid) -> vtk.vtkPolyData:
    """Extract external boundary surface and keep original bulk ids if possible."""
    surf = vtk.vtkDataSetSurfaceFilter()
    surf.SetInputData(ug)

    # These methods exist in common VTK versions. Keep guarded for compatibility.
    if hasattr(surf, "PassThroughPointIdsOn"):
        surf.PassThroughPointIdsOn()
    if hasattr(surf, "PassThroughCellIdsOn"):
        surf.PassThroughCellIdsOn()
    if hasattr(surf, "SetOriginalPointIdsName"):
        surf.SetOriginalPointIdsName("bulk_node_ids")
    if hasattr(surf, "SetOriginalCellIdsName"):
        surf.SetOriginalCellIdsName("bulk_cell_ids")

    surf.Update()
    boundary = surf.GetOutput()
    if boundary is None or boundary.GetNumberOfCells() == 0:
        raise RuntimeError("Boundary extraction produced an empty surface.")
    return boundary


def compute_cell_normals(boundary: vtk.vtkPolyData) -> vtk.vtkPolyData:
    norm_filter = vtk.vtkPolyDataNormals()
    norm_filter.SetInputData(boundary)
    norm_filter.ComputePointNormalsOff()
    norm_filter.ComputeCellNormalsOn()
    norm_filter.SplittingOff()
    norm_filter.ConsistencyOn()
    norm_filter.Update()
    out = norm_filter.GetOutput()
    normals = out.GetCellData().GetNormals()
    if normals is None:
        raise RuntimeError("Failed to compute cell normals on boundary surface.")
    return out


def cell_centroid(cell: vtk.vtkCell, points: vtk.vtkPoints) -> tuple[float, float, float]:
    num_pts = cell.GetNumberOfPoints()
    cx = cy = cz = 0.0
    for i in range(num_pts):
        pid = cell.GetPointId(i)
        x, y, z = points.GetPoint(pid)
        cx += x
        cy += y
        cz += z
    return cx / num_pts, cy / num_pts, cz / num_pts


def select_cells_as_unstructured_grid(
    dataset: vtk.vtkDataSet,
    cell_ids: list[int],
) -> vtk.vtkUnstructuredGrid:
    ids = vtk.vtkIdTypeArray()
    ids.SetNumberOfComponents(1)
    ids.SetName("SelectedBoundaryCellIds")
    for cid in cell_ids:
        ids.InsertNextValue(int(cid))

    selection_node = vtk.vtkSelectionNode()
    selection_node.SetFieldType(vtk.vtkSelectionNode.CELL)
    selection_node.SetContentType(vtk.vtkSelectionNode.INDICES)
    selection_node.SetSelectionList(ids)

    selection = vtk.vtkSelection()
    selection.AddNode(selection_node)

    extract = vtk.vtkExtractSelection()
    extract.SetInputData(0, dataset)
    extract.SetInputData(1, selection)
    extract.Update()

    out = vtk.vtkUnstructuredGrid.SafeDownCast(extract.GetOutput())
    if out is None:
        # Fallback for unusual VTK builds.
        append = vtk.vtkAppendFilter()
        append.AddInputData(extract.GetOutput())
        append.Update()
        out = append.GetOutput()

    if out is None or out.GetNumberOfCells() == 0:
        raise RuntimeError("Selected face is empty after extraction.")
    return out


def ensure_bulk_node_ids_uint64(face: vtk.vtkUnstructuredGrid) -> None:
    """
    Ensure the face mesh has UInt64 point-data array named bulk_node_ids.

    OGS boundary meshes often need this array. If original point ids were
    preserved by vtkDataSetSurfaceFilter, use them. Otherwise fall back to local
    point ids and print a warning.
    """
    pd = face.GetPointData()

    source = pd.GetArray("bulk_node_ids")
    if source is None:
        source = pd.GetArray("vtkOriginalPointIds")

    out = vtk.vtkUnsignedLongLongArray()
    out.SetName("bulk_node_ids")
    out.SetNumberOfComponents(1)
    out.SetNumberOfTuples(face.GetNumberOfPoints())

    if source is not None:
        for i in range(face.GetNumberOfPoints()):
            out.SetValue(i, int(source.GetTuple1(i)))
    else:
        print(
            "Warning: original bulk point ids were not found; "
            "using local point ids for bulk_node_ids."
        )
        for i in range(face.GetNumberOfPoints()):
            out.SetValue(i, int(i))

    # Remove old arrays with the same name to avoid duplicate-name ambiguity.
    if pd.GetArray("bulk_node_ids") is not None:
        pd.RemoveArray("bulk_node_ids")
    pd.AddArray(out)


def write_unstructured_grid(ug: vtk.vtkUnstructuredGrid, path: str | Path, binary: bool = True) -> None:
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(ug)
    if binary:
        writer.SetDataModeToBinary()
    else:
        writer.SetDataModeToAscii()
    ok = writer.Write()
    if ok != 1:
        raise RuntimeError(f"Failed to write VTU file: {path}")


def group_boundary_cells(
    boundary: vtk.vtkPolyData,
    bounds: tuple[float, float, float, float, float, float],
    coord_tol: float,
    axis_tol: float,
    off_tol: float,
) -> dict[str, list[int]]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    points = boundary.GetPoints()
    normals = boundary.GetCellData().GetNormals()
    num_cells = boundary.GetNumberOfCells()

    groups: dict[str, list[int]] = {name: [] for name in FACE_NAMES}

    for cid in range(num_cells):
        cell = boundary.GetCell(cid)
        cx, cy, cz = cell_centroid(cell, points)
        nx, ny, nz = normals.GetTuple(cid)

        is_x = abs(nx) > axis_tol and abs(ny) < off_tol and abs(nz) < off_tol
        is_y = abs(ny) > axis_tol and abs(nx) < off_tol and abs(nz) < off_tol
        is_z = abs(nz) > axis_tol and abs(nx) < off_tol and abs(ny) < off_tol

        if is_x and abs(cx - xmin) <= coord_tol:
            groups["x_min"].append(cid)
        elif is_x and abs(cx - xmax) <= coord_tol:
            groups["x_max"].append(cid)
        elif is_y and abs(cy - ymin) <= coord_tol:
            groups["y_min"].append(cid)
        elif is_y and abs(cy - ymax) <= coord_tol:
            groups["y_max"].append(cid)
        elif is_z and abs(cz - zmin) <= coord_tol:
            groups["z_min"].append(cid)
        elif is_z and abs(cz - zmax) <= coord_tol:
            groups["z_max"].append(cid)

    return groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract six boundary faces from a VTU mesh and write VTU files."
    )
    parser.add_argument(
        "--input",
        default="reservoir_with_lowres.vtu",
        help="Input VTU mesh. Default: reservoir_with_lowres.vtu",
    )
    parser.add_argument(
        "--output-dir",
        default="six_faces_vtu",
        help="Output directory. Default: six_faces_vtu",
    )
    parser.add_argument(
        "--prefix",
        default="yangyi_face",
        help="Output file prefix. Default: yangyi_face",
    )
    parser.add_argument(
        "--coord-tol",
        type=float,
        default=None,
        help="Coordinate tolerance. Default: model size * 1e-6",
    )
    parser.add_argument(
        "--axis-tol",
        type=float,
        default=0.95,
        help="Normal-axis threshold. Default: 0.95",
    )
    parser.add_argument(
        "--off-tol",
        type=float,
        default=0.15,
        help="Normal off-axis threshold. Default: 0.15",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Write ASCII VTU instead of binary.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading input mesh:", input_path)
    ug = read_unstructured_grid(input_path)
    bounds = ug.GetBounds()
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    print("Bounds:", bounds)
    print("Input cells:", ug.GetNumberOfCells())
    print("Input points:", ug.GetNumberOfPoints())

    print("Extracting external boundary...")
    boundary = extract_boundary_with_original_ids(ug)
    boundary = compute_cell_normals(boundary)
    print("Boundary cells:", boundary.GetNumberOfCells())
    print("Boundary points:", boundary.GetNumberOfPoints())

    model_size = max(xmax - xmin, ymax - ymin, zmax - zmin)
    coord_tol = args.coord_tol if args.coord_tol is not None else model_size * 1.0e-6
    print("Coordinate tolerance:", coord_tol)

    groups = group_boundary_cells(
        boundary=boundary,
        bounds=bounds,
        coord_tol=coord_tol,
        axis_tol=args.axis_tol,
        off_tol=args.off_tol,
    )

    print("\nFace extraction summary:")
    for name in FACE_NAMES:
        cids = groups[name]
        if not cids:
            print(f"  Warning: no cells assigned to {name}")
            continue

        face = select_cells_as_unstructured_grid(boundary, cids)
        ensure_bulk_node_ids_uint64(face)

        out_file = output_dir / f"{args.prefix}_{name}.vtu"
        write_unstructured_grid(face, out_file, binary=not args.ascii)

        print(
            f"  {name}: {len(cids)} selected cells -> {out_file} "
            f"({face.GetNumberOfCells()} cells, {face.GetNumberOfPoints()} points)"
        )

    print("\nSix-face VTU extraction complete.")


if __name__ == "__main__":
    main()
