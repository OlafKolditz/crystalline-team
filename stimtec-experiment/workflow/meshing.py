"""Mesh preparation helpers for the STIMTEC DFN workflow."""

import os
import shutil
from pathlib import Path

import pyvista as pv

from workflow_paths import find_borehole_seed_file, find_mesh_file


MESH_FILENAME = "mixed_dimensional_all.vtu"
BOREHOLE_SEED_FILENAME = "BH10.vtu"


def get_out_dir(out_dir: Path | None = None, base_dir: Path | None = None) -> Path:
    """Resolve and create the workflow output directory."""
    if out_dir is None:
        default_out_dir = "_out" if base_dir is None else str(Path(base_dir) / "_out")
        out_dir = Path(os.environ.get("OGS_TESTRUNNER_OUT_DIR", default_out_dir))

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def copy_meshes_to_output(
    mesh_file: Path,
    borehole_seed_file: Path,
    out_dir: Path,
    mesh_output_name: str | None = None,
    borehole_output_name: str | None = None,
) -> tuple[Path, Path]:
    """Copy the source meshes into the output workspace."""
    dst_file = out_dir / (mesh_output_name or mesh_file.name)
    dst_bh10 = out_dir / (borehole_output_name or borehole_seed_file.name)

    for src, dst in ((mesh_file, dst_file), (borehole_seed_file, dst_bh10)):
        if not src.exists():
            raise FileNotFoundError(f"Required mesh file not found: {src}")
        if src.resolve() == dst.resolve():
            continue
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.copy2(src, dst)

    return dst_file, dst_bh10


def clean_mesh(mesh_file: Path) -> pv.DataSet:
    """Remove duplicated mesh points before the OGS preprocessing steps."""
    mesh = pv.read(mesh_file)
    cleaned_mesh = mesh.clean(tolerance=1e-12)
    cleaned_mesh.save(mesh_file)

    print(f"Original points: {mesh.n_points}, Cleaned points: {cleaned_mesh.n_points}")
    return cleaned_mesh


def extract_boundary_planes(cleaned_mesh: pv.DataSet) -> tuple[float, float, float, float, float, float]:
    """Compute boundary planes slightly inside the model bounds."""
    xmin, xmax, ymin, ymax, zmin, zmax = cleaned_mesh.bounds
    lx, ly, lz = xmax - xmin, ymax - ymin, zmax - zmin
    domain_size = max(lx, ly, lz)

    boundary_tol_fraction = 1e-10
    tol = boundary_tol_fraction * domain_size

    x_left = xmin + tol
    x_right = xmax - tol
    y_front = ymin + tol
    y_back = ymax - tol
    z_bottom = zmin + tol
    z_top = zmax - tol

    print("Bounds:", cleaned_mesh.bounds)
    print("tol =", tol)

    return x_left, x_right, y_front, y_back, z_bottom, z_top


def extract_boundary_mesh_and_planes(
    out_dir: Path,
    mesh_file: Path,
    x_left: float,
    x_right: float,
    y_front: float,
    y_back: float,
    z_bottom: float,
    z_top: float,
) -> None:
    """Create the boundary condition meshes used by the OGS setup."""
    import ogstools as ot

    ot.cli().NodeReordering(
        i=str(mesh_file),
        o=str(mesh_file),
        m=0,
        no_volume_check=True,
    )

    outer_boundary = out_dir / "outer_boundary.vtu"
    ot.cli().ExtractBoundary(i=str(mesh_file), o=str(outer_boundary))

    ot.cli().removeMeshElements(
        i=str(outer_boundary),
        o=str(out_dir / "bc_xmin.vtu"),
        **{"x-max": x_left, "invert": True},
    )
    ot.cli().removeMeshElements(
        i=str(outer_boundary),
        o=str(out_dir / "bc_xmax.vtu"),
        **{"x-min": x_right, "invert": True},
    )
    ot.cli().removeMeshElements(
        i=str(outer_boundary),
        o=str(out_dir / "bc_ymin.vtu"),
        **{"y-max": y_front, "invert": True},
    )
    ot.cli().removeMeshElements(
        i=str(outer_boundary),
        o=str(out_dir / "bc_ymax.vtu"),
        **{"y-min": y_back, "invert": True},
    )
    ot.cli().removeMeshElements(
        i=str(outer_boundary),
        o=str(out_dir / "bc_zmin.vtu"),
        **{"z-max": z_bottom, "invert": True},
    )
    ot.cli().removeMeshElements(
        i=str(outer_boundary),
        o=str(out_dir / "bc_zmax.vtu"),
        **{"z-min": z_top, "invert": True},
    )


def identify_subdomains(out_dir: Path, mesh_filename: str, borehole_seed_filename: str) -> None:
    """Assign subdomain ids from the prepared seed meshes."""
    import ogstools as ot

    seed_meshes = [
        "bc_xmin.vtu",
        "bc_xmax.vtu",
        "bc_ymin.vtu",
        "bc_ymax.vtu",
        "bc_zmin.vtu",
        "bc_zmax.vtu",
        borehole_seed_filename,
    ]

    cwd_before_identify = Path.cwd()
    os.chdir(out_dir)
    try:
        ot.cli().identifySubdomains(
            "-m",
            mesh_filename,
            "--",
            *seed_meshes,
        )
    finally:
        os.chdir(cwd_before_identify)


def run_meshing(
    orig_dir: Path | None = None,
    out_dir: Path | None = None,
    mesh_file: Path | None = None,
    borehole_seed_file: Path | None = None,
    mesh_output_name: str | None = None,
    borehole_output_name: str | None = None,
) -> Path:
    """Run the full meshing and mesh-preparation workflow."""
    orig_dir = Path.cwd().resolve() if orig_dir is None else Path(orig_dir).resolve()
    out_dir = get_out_dir(out_dir, base_dir=orig_dir)

    if mesh_file is None:
        mesh_file = find_mesh_file(orig_dir)
    if borehole_seed_file is None:
        borehole_seed_file = find_borehole_seed_file(orig_dir)

    if mesh_file is None:
        raise FileNotFoundError(
            f"No mesh file found under {orig_dir}. Expected one of ('mixed_dimensional_all.vtu', 'mesh.vtu')."
        )
    if borehole_seed_file is None:
        raise FileNotFoundError(f"No borehole seed file found under {orig_dir}. Expected BH10.vtu.")

    prepared_mesh_file, prepared_borehole_seed_file = copy_meshes_to_output(
        mesh_file=Path(mesh_file),
        borehole_seed_file=Path(borehole_seed_file),
        out_dir=out_dir,
        mesh_output_name=mesh_output_name,
        borehole_output_name=borehole_output_name,
    )

    cleaned_mesh = clean_mesh(prepared_mesh_file)
    planes = extract_boundary_planes(cleaned_mesh)
    extract_boundary_mesh_and_planes(out_dir, prepared_mesh_file, *planes)
    identify_subdomains(out_dir, prepared_mesh_file.name, prepared_borehole_seed_file.name)

    print("Meshing completed!")
    return out_dir
