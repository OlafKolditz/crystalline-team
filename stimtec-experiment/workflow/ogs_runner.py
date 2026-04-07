"""Helpers for running the OGS model."""

from pathlib import Path

from workflow_paths import find_project_file, read_project_mesh_names, read_project_output_prefix


PROJECT_FILENAME = "STIMTEC_DFN.prj"
PVD_FILENAME = "Stimtec_DFN.pvd"


def run_ogs(project_file: Path, out_dir: Path) -> None:
    """Write the final project file and execute the OGS model."""
    import ogstools as ot

    output_project_file = out_dir / f"{project_file.stem}_final.prj"
    project = ot.Project(
        input_file=project_file,
        output_file=output_project_file,
    )
    project.write_input()
    project.run_model(args=f"-o {out_dir} -m {out_dir}", logfile=out_dir / "run.log")


def _validate_project_meshes(project_file: Path, mesh_dir: Path) -> None:
    missing_meshes = [mesh_name for mesh_name in read_project_mesh_names(project_file) if not (mesh_dir / mesh_name).exists()]
    if missing_meshes:
        raise FileNotFoundError(
            f"Project {project_file} expects mesh files in {mesh_dir}, but these are missing: {missing_meshes}"
        )


def run_ogs_simulation(orig_dir: Path, out_dir: Path, project_file: Path | None = None) -> Path:
    """Run the OGS simulation after meshing has prepared the workspace."""
    orig_dir = Path(orig_dir).resolve()
    out_dir = Path(out_dir).resolve()

    if project_file is None:
        project_file = find_project_file(orig_dir)
    if project_file is None:
        raise FileNotFoundError(
            f"No project file found under {orig_dir}. Expected one of ('STIMTEC_DFN.prj', 'model01.prj')."
        )

    project_file = Path(project_file).resolve()
    _validate_project_meshes(project_file, out_dir)
    run_ogs(project_file=project_file, out_dir=out_dir)

    output_prefix = read_project_output_prefix(project_file) or Path(PVD_FILENAME).stem

    print("OGS simulation completed!")
    return out_dir / f"{output_prefix}.pvd"
