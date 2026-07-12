#!/usr/bin/env python3
"""
Estimate PorePy DFN fault parameters from fault surface .vtp files.

Purpose
-------
This script reads one or more VTP fault-surface files and estimates the
parameters needed by generate_yangyi_dfn.py:

    fault_id, center_x, center_y, center_z,
    major_axis, minor_axis, major_axis_angle,
    strike, dip, material_id

Assumptions
-----------
1. Each .vtp file represents one fault surface.
2. Coordinates use the convention:
      x = East, y = North, z = Up
3. strike is reported clockwise from North, in degrees, in [0, 180).
4. dip is reported in degrees, in [0, 90].
5. major_axis and minor_axis are semi-axis lengths, not full lengths.
6. major_axis_angle is the in-plane angle from strike direction to the major axis.

Outputs
-------
1. faults_all_from_vtp.csv
   PorePy-ready table. This can be renamed to faults_all.csv.

2. faults_estimated_diagnostics.csv
   Same parameters plus diagnostic columns.

Usage
-----
Run in the folder containing the .vtp files:

    python estimate_fault_parameters_from_vtp.py

Or edit INPUT_DIR below.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pyvista as pv


# ============================================================
# User settings
# ============================================================
# Folder containing the VTP fault files.
# Use Path(".") if running this script inside the VTP folder.
INPUT_DIR = Path(".")

# Output files
OUTPUT_POREPY_CSV = "faults_all_from_vtp.csv"
OUTPUT_DIAGNOSTIC_CSV = "faults_estimated_diagnostics.csv"

# Default material ID assigned to all faults.
# You can edit material_id manually in the output CSV later.
DEFAULT_MATERIAL_ID = 1

# File pattern. Normally keep this as *.vtp.
FILE_PATTERN = "*.vtp"

# Minimum required points for fitting a plane.
MIN_POINTS = 3

# Optional: remove duplicate points before fitting.
REMOVE_DUPLICATE_POINTS = True

# Number of decimals in output CSV.
FLOAT_FORMAT = "%.6f"


# ============================================================
# Vector utilities
# ============================================================
def normalize(vector: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Return a unit vector."""
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def unique_points(points: np.ndarray, decimals: int = 10) -> np.ndarray:
    """Remove duplicate points using rounded coordinates."""
    rounded = np.round(points, decimals=decimals)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return points[unique_indices]


# ============================================================
# Geometry calculations
# ============================================================
def compute_strike_dip_from_normal(normal: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Compute geological strike and dip from a plane normal.

    Coordinate convention:
        x = East, y = North, z = Up

    strike:
        clockwise from North, degrees, range [0, 180)

    dip:
        degrees, range [0, 90]
    """
    n = normalize(normal)

    # Use upward-pointing normal for stable convention.
    if n[2] < 0:
        n = -n

    # Dip: plane angle relative to horizontal.
    # Horizontal plane -> normal vertical -> dip = 0.
    # Vertical plane -> normal horizontal -> dip = 90.
    dip = np.degrees(np.arccos(np.clip(abs(n[2]), 0.0, 1.0)))

    # Strike direction is the intersection between the fault plane and horizontal plane.
    # It is perpendicular to both vertical axis and the plane normal.
    z_axis = np.array([0.0, 0.0, 1.0])
    strike_vec = np.cross(z_axis, n)

    if np.linalg.norm(strike_vec) < 1e-12:
        # Strike is undefined for nearly horizontal planes.
        strike = 0.0
    else:
        strike_vec = normalize(strike_vec)
        # Convert vector components (east, north) to azimuth clockwise from North.
        strike = np.degrees(np.arctan2(strike_vec[0], strike_vec[1]))
        if strike < 0.0:
            strike += 360.0
        # Strike has 180-degree ambiguity.
        if strike >= 180.0:
            strike -= 180.0

    return float(strike), float(dip), n


def strike_and_dip_vectors(strike_deg: float, dip_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return strike vector and down-dip vector using the same convention as
    generate_yangyi_dfn.py.
    """
    strike_rad = np.deg2rad(strike_deg)
    dip_rad = np.deg2rad(dip_deg)

    # Horizontal strike direction.
    strike_vec = np.array(
        [
            np.sin(strike_rad),
            np.cos(strike_rad),
            0.0,
        ],
        dtype=float,
    )
    strike_vec = normalize(strike_vec)

    # Down-dip direction.
    dip_vec = np.array(
        [
            np.cos(strike_rad) * np.cos(dip_rad),
            -np.sin(strike_rad) * np.cos(dip_rad),
            -np.sin(dip_rad),
        ],
        dtype=float,
    )
    dip_vec = normalize(dip_vec)

    return strike_vec, dip_vec


def estimate_fault_parameters(mesh_file: Path, material_id: int = DEFAULT_MATERIAL_ID) -> dict:
    """Estimate DFN parameters from one VTP fault surface."""
    mesh = pv.read(mesh_file)

    if mesh.n_points < MIN_POINTS:
        raise ValueError(f"{mesh_file}: not enough points to fit a plane.")

    points = np.asarray(mesh.points, dtype=float)

    if REMOVE_DUPLICATE_POINTS:
        points = unique_points(points)

    if points.shape[0] < MIN_POINTS:
        raise ValueError(f"{mesh_file}: not enough unique points to fit a plane.")

    # Center of all points.
    center = points.mean(axis=0)

    # Principal component analysis using SVD.
    # The first two components span the best-fit plane.
    # The third component is the plane normal.
    centered = points - center
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    major_dir = normalize(vt[0])
    minor_dir = normalize(vt[1])
    normal = normalize(vt[2])

    # Enforce right-handed local coordinate system.
    if np.dot(np.cross(major_dir, minor_dir), normal) < 0.0:
        minor_dir = -minor_dir

    strike, dip, normal_up = compute_strike_dip_from_normal(normal)
    strike_vec, dip_vec = strike_and_dip_vectors(strike, dip)

    # Project all points onto the in-plane PCA directions.
    coord_major = centered @ major_dir
    coord_minor = centered @ minor_dir

    # Semi-axis lengths estimated as half the spatial range in each direction.
    major_axis = 0.5 * (coord_major.max() - coord_major.min())
    minor_axis = 0.5 * (coord_minor.max() - coord_minor.min())

    # Ensure major_axis >= minor_axis.
    if minor_axis > major_axis:
        major_axis, minor_axis = minor_axis, major_axis
        major_dir, minor_dir = minor_dir, major_dir

    # Major axis angle relative to strike direction within the fault plane.
    # Positive direction is toward down-dip vector.
    a = float(np.dot(major_dir, strike_vec))
    b = float(np.dot(major_dir, dip_vec))
    major_axis_angle = np.degrees(np.arctan2(b, a))

    # Because the major axis has 180-degree ambiguity, normalize to [0, 180).
    if major_axis_angle < 0.0:
        major_axis_angle += 180.0
    if major_axis_angle >= 180.0:
        major_axis_angle -= 180.0

    # Diagnostics: planarity from SVD.
    # Smaller value means points lie closer to a plane.
    if singular_values[0] > 0.0:
        planarity_ratio = singular_values[2] / singular_values[0]
    else:
        planarity_ratio = np.nan

    # Approximate area from triangulated mesh if possible.
    area = np.nan
    try:
        surface = mesh.extract_surface().triangulate()
        area = float(surface.area)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Could not compute area for {mesh_file}: {exc}")

    return {
        "fault_id": mesh_file.stem,
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "major_axis": float(major_axis),
        "minor_axis": float(minor_axis),
        "major_axis_angle": float(major_axis_angle),
        "strike": float(strike),
        "dip": float(dip),
        "material_id": int(material_id),
        # Diagnostic columns
        "n_points_original": int(mesh.n_points),
        "n_points_unique": int(points.shape[0]),
        "n_cells": int(mesh.n_cells),
        "sv1": float(singular_values[0]),
        "sv2": float(singular_values[1]),
        "sv3": float(singular_values[2]),
        "planarity_ratio_sv3_over_sv1": float(planarity_ratio),
        "surface_area": area,
        "x_min": float(points[:, 0].min()),
        "x_max": float(points[:, 0].max()),
        "y_min": float(points[:, 1].min()),
        "y_max": float(points[:, 1].max()),
        "z_min": float(points[:, 2].min()),
        "z_max": float(points[:, 2].max()),
    }


# ============================================================
# Main workflow
# ============================================================
def main() -> None:
    input_dir = INPUT_DIR.resolve()
    fault_files = sorted(input_dir.glob(FILE_PATTERN))

    if not fault_files:
        raise FileNotFoundError(
            f"No files matching {FILE_PATTERN!r} found in {input_dir}. "
            "Put this script in the VTP folder or edit INPUT_DIR."
        )

    print(f"Input directory: {input_dir}")
    print(f"Number of VTP files: {len(fault_files)}")

    rows = []
    for mesh_file in fault_files:
        print(f"Processing: {mesh_file.name}")
        rows.append(estimate_fault_parameters(mesh_file, DEFAULT_MATERIAL_ID))

    df = pd.DataFrame(rows)

    porepy_columns = [
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

    diagnostic_columns = porepy_columns + [
        "n_points_original",
        "n_points_unique",
        "n_cells",
        "sv1",
        "sv2",
        "sv3",
        "planarity_ratio_sv3_over_sv1",
        "surface_area",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "z_min",
        "z_max",
    ]

    df[porepy_columns].to_csv(OUTPUT_POREPY_CSV, index=False, float_format=FLOAT_FORMAT)
    df[diagnostic_columns].to_csv(
        OUTPUT_DIAGNOSTIC_CSV, index=False, float_format=FLOAT_FORMAT
    )

    print("\nSaved:")
    print(f"  {OUTPUT_POREPY_CSV}")
    print(f"  {OUTPUT_DIAGNOSTIC_CSV}")

    print("\nPorePy-ready preview:")
    print(df[porepy_columns].to_string(index=False))

    print("\nNotes:")
    print("  - major_axis and minor_axis are semi-axis lengths, not full lengths.")
    print("  - material_id is currently assigned as DEFAULT_MATERIAL_ID for all faults.")
    print("  - After checking, rename faults_all_from_vtp.csv to faults_all.csv if needed.")
    print("  - Check strike/dip against your geological interpretation before final meshing.")


if __name__ == "__main__":
    main()
