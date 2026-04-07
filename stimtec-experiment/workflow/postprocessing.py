"""Postprocessing helpers for the STIMTEC DFN workflow."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from workflow_paths import find_any_pvd_file, find_mesh_file

try:
    from IPython.display import Image as IPyImage
    from IPython.display import Markdown, display
except ImportError:
    IPyImage = None
    Markdown = None
    display = print


pv.OFF_SCREEN = True

INITIAL_MESH_FILENAME = "mixed_dimensional_all.vtu"
PVD_FILENAME = "Stimtec_DFN.pvd"
OBSERVATION_POINT = np.array([4.59509e6, 5.6442e6, 268.933], dtype=float)
PRESSURE_OBSERVATION_FIGURE = "pressure_observation_point.png"

base_scalar_bar_args = {
    "vertical": True,
    "title_font_size": 18,
    "label_font_size": 14,
    "shadow": False,
    "n_labels": 5,
    "fmt": "%.2e",
}


def save_scalar_png_safe(
    mesh,
    scalar_name: str,
    title: str,
    png_path: str | Path,
    cmap: str = "Blues",
    opacity: float = 1.0,
    clip: bool = False,
    clip_normal: tuple[float, float, float] = (1.0, 0.0, 0.0),
    show_edges: bool = False,
    clim: tuple[float, float] | None = None,
    n_colors: int | None = None,
) -> None:
    """Safely render a scalar field to a PNG."""
    png_path = Path(png_path)

    if not isinstance(mesh, pv.DataSet):
        try:
            mesh = pv.wrap(mesh)
        except Exception as exc:
            raise TypeError(
                f"mesh is not a PyVista dataset and cannot be wrapped. type={type(mesh)}"
            ) from exc

    has_point = scalar_name in mesh.point_data
    has_cell = scalar_name in mesh.cell_data
    if not (has_point or has_cell):
        raise KeyError(
            f"Scalar '{scalar_name}' not found.\n"
            f"point_data keys: {list(mesh.point_data.keys())}\n"
            f"cell_data keys : {list(mesh.cell_data.keys())}"
        )

    if clip:
        mesh = mesh.clip(normal=clip_normal)

    preference = "cell" if has_cell and not has_point else "point"

    plotter = pv.Plotter(off_screen=True, window_size=[1600, 1000])

    scalar_bar_args = dict(base_scalar_bar_args)
    scalar_bar_args["title"] = title

    add_mesh_kwargs = dict(
        scalars=scalar_name,
        preference=preference,
        cmap=cmap,
        show_edges=show_edges,
        opacity=opacity,
        scalar_bar_args=scalar_bar_args,
    )
    if clim is not None:
        add_mesh_kwargs["clim"] = clim
    if n_colors is not None:
        add_mesh_kwargs["n_colors"] = n_colors

    plotter.add_mesh(mesh, **add_mesh_kwargs)
    plotter.add_mesh(mesh.outline(), color="black", line_width=2)
    plotter.show_axes()
    plotter.enable_parallel_projection()
    plotter.view_isometric()

    png_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(png_path))
    plotter.close()

    print("Saved:", png_path)
    if Markdown:
        display(Markdown(f"**{title}**"))
    if IPyImage:
        display(IPyImage(filename=str(png_path)))


VELOCITY_VECTOR_NAMES = ("v", "darcy_velocity", "velocity")
MAX_VELOCITY_ARROWS = 900


def _mesh_with_point_field(mesh, field_name: str):
    if field_name in mesh.point_data:
        return mesh
    if field_name in mesh.cell_data:
        converted = mesh.cell_data_to_point_data(pass_cell_data=True)
        if field_name in converted.point_data:
            return converted
    raise KeyError(
        f"Field '{field_name}' not found.\n"
        f"point_data keys: {list(mesh.point_data.keys())}\n"
        f"cell_data keys : {list(mesh.cell_data.keys())}"
    )


def _resolve_velocity_field(mesh):
    for field_name in VELOCITY_VECTOR_NAMES:
        if field_name in mesh.point_data:
            return mesh, field_name
        if field_name in mesh.cell_data:
            converted = mesh.cell_data_to_point_data(pass_cell_data=True)
            if field_name in converted.point_data:
                return converted, field_name
    raise KeyError(
        "No velocity vector field found.\n"
        f"Expected one of {VELOCITY_VECTOR_NAMES}.\n"
        f"point_data keys: {list(mesh.point_data.keys())}\n"
        f"cell_data keys : {list(mesh.cell_data.keys())}"
    )


def _build_velocity_glyphs(mesh, vector_name: str, max_arrows: int = MAX_VELOCITY_ARROWS):
    vectors = np.asarray(mesh.point_data[vector_name], dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"Velocity field '{vector_name}' is not a 3-component vector array.")

    magnitude = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(magnitude) & np.all(np.isfinite(vectors), axis=1) & (magnitude > 0.0)
    if not np.any(valid):
        raise ValueError(f"Velocity field '{vector_name}' does not contain positive finite vectors.")

    points = np.asarray(mesh.points)[valid]
    vectors = vectors[valid]
    magnitude = magnitude[valid]

    if len(points) > max_arrows:
        step = int(np.ceil(len(points) / max_arrows))
        indices = np.arange(0, len(points), step)
        points = points[indices]
        vectors = vectors[indices]
        magnitude = magnitude[indices]

    glyph_input = pv.PolyData(points)
    glyph_input[vector_name] = vectors
    glyph_input["velocity_magnitude"] = magnitude

    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    domain_size = max(xmax - xmin, ymax - ymin, zmax - zmin)
    vmax = float(magnitude.max())
    scale_factor = domain_size * 0.10 / vmax if vmax > 0.0 else 1.0

    return glyph_input.glyph(
        scale="velocity_magnitude",
        orient=vector_name,
        factor=scale_factor,
    )


def save_pressure_profile_with_velocity(mesh, png_path: str | Path) -> None:
    """Render the pressure field and overlay velocity glyphs."""
    png_path = Path(png_path)

    mesh = _mesh_with_point_field(mesh, "pressure")
    mesh, vector_name = _resolve_velocity_field(mesh)
    arrows = _build_velocity_glyphs(mesh, vector_name)

    scalar_bar_args = dict(base_scalar_bar_args)
    scalar_bar_args["title"] = "Pressure [Pa]"

    plotter = pv.Plotter(off_screen=True, window_size=[1600, 1000])
    plotter.add_mesh(
        mesh,
        scalars="pressure",
        preference="point",
        cmap="Blues",
        opacity=0.15,
        show_edges=False,
        scalar_bar_args=scalar_bar_args,
    )
    plotter.add_mesh(arrows, color="black")
    plotter.add_mesh(mesh.outline(), color="black", line_width=2)
    plotter.show_axes()
    plotter.enable_parallel_projection()
    plotter.view_isometric()

    png_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(png_path))
    plotter.close()

    print("Saved:", png_path)
    if Markdown:
        display(Markdown("**Pressure Profile With Darcy Velocity**"))
    if IPyImage:
        display(IPyImage(filename=str(png_path)))


def extract_fracture_mesh(mesh, fracture_ids=(1, 2, 3, 4)):
    if "MaterialIDs" not in mesh.cell_data:
        raise KeyError(
            f"'MaterialIDs' not found in mesh cell_data. Available keys: {list(mesh.cell_data.keys())}"
        )

    fracture_ids = np.asarray(fracture_ids, dtype=int)
    fracture_mask = np.isin(np.asarray(mesh.cell_data["MaterialIDs"]).astype(int), fracture_ids)
    if not np.any(fracture_mask):
        raise ValueError(f"No fracture cells with MaterialIDs {fracture_ids.tolist()} were found in the mesh.")
    return mesh.extract_cells(fracture_mask)


def save_ogstools_contourf_png(mesh, variable, png_path: str | Path, title: str, **kwargs) -> None:
    import ogstools as ot

    png_path = Path(png_path)
    fig = ot.plot.contourf(mesh, variable, **kwargs)
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", png_path)
    if Markdown:
        display(Markdown(f"**{title}**"))
    if IPyImage:
        display(IPyImage(filename=str(png_path)))


def save_velocity_contourf_png(mesh, png_path: str | Path) -> None:
    import ogstools as ot

    fracture_mesh = extract_fracture_mesh(mesh)
    if "v" in fracture_mesh.point_data:
        if "velocity" not in fracture_mesh.point_data:
            fracture_mesh.point_data["velocity"] = fracture_mesh.point_data["v"]
        if "darcy_velocity" not in fracture_mesh.point_data:
            fracture_mesh.point_data["darcy_velocity"] = fracture_mesh.point_data["v"]

    try:
        save_ogstools_contourf_png(
            fracture_mesh,
            ot.variables.velocity,
            png_path=png_path,
            title="Fracture Darcy Velocity",
            show_region_bounds=False,
            show_edges=False,
            log_scaled=True,
            vmin=-8,
            figsize=(12, 8),
        )
    except Exception as exc:
        print(f"Falling back to PyVista velocity-magnitude rendering because ogstools contouring failed: {exc}")
        fracture_mesh, vector_name = _resolve_velocity_field(fracture_mesh)
        velocity_magnitude = np.linalg.norm(np.asarray(fracture_mesh.point_data[vector_name], dtype=float), axis=1)
        fracture_mesh.point_data["velocity_magnitude"] = np.where(np.isfinite(velocity_magnitude), velocity_magnitude, 0.0)

        save_scalar_png_safe(
            fracture_mesh,
            scalar_name="velocity_magnitude",
            title="Fracture Darcy Velocity Magnitude",
            png_path=png_path,
            cmap="viridis",
            opacity=1.0,
            clip=False,
            show_edges=False,
        )


def save_pressure_observation_png(
    mesh_series,
    point: np.ndarray,
    png_path: str | Path,
) -> None:
    import ogstools as ot

    png_path = Path(png_path)
    pressures = np.squeeze(np.asarray(mesh_series.probe(point[np.newaxis, :], ot.variables.pressure)))
    times = np.asarray(mesh_series.timevalues, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, pressures, color=ot.variables.pressure.color, linewidth=2.5)
    ax.scatter(times[-1], pressures[-1], color=ot.variables.pressure.color, s=50, zorder=3)
    ax.set_xlabel("time / s")
    ax.set_ylabel(ot.variables.pressure.get_label())
    ax.set_title(
        "Pressure At Observation Point\n"
        f"x={point[0]:.3f} m, y={point[1]:.3f} m, z={point[2]:.3f} m"
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", png_path)
    if Markdown:
        display(Markdown("**Pressure At Observation Point**"))
    if IPyImage:
        display(IPyImage(filename=str(png_path)))


def save_material_ids_png(mesh, png_path: str | Path) -> None:
    if "MaterialIDs" not in mesh.cell_data:
        raise KeyError(
            f"'MaterialIDs' not found in mesh cell_data. Available keys: {list(mesh.cell_data.keys())}"
        )

    material_ids = np.unique(np.asarray(mesh.cell_data["MaterialIDs"]).astype(int))
    print("MaterialIDs in initial mesh:", material_ids.tolist())

    mesh = extract_fracture_mesh(mesh)
    save_scalar_png_safe(
        mesh,
        scalar_name="MaterialIDs",
        title="Fracture MaterialIDs (1-4)",
        png_path=png_path,
        cmap="tab20",
        opacity=1.0,
        clip=False,
        show_edges=True,
        clim=(0.5, 4.5),
        n_colors=4,
    )


def save_initial_material_ids_png(out_dir: Path, mesh_file: str | Path | None = None) -> None:
    """Render the initial fracture material-id plot from the prepared mesh."""
    mesh_file = Path(mesh_file) if mesh_file is not None else find_mesh_file(out_dir)
    if mesh_file is None:
        raise FileNotFoundError(f"No initial mesh file found under {out_dir}.")
    if not mesh_file.exists():
        raise FileNotFoundError(f"Initial mesh file not found: {mesh_file}")

    initial_mesh = pv.read(mesh_file)
    save_material_ids_png(initial_mesh, out_dir / "material_ids_initial.png")


def run_postprocessing(
    out_dir: Path,
    pvd_file: str | Path | None = None,
    initial_mesh_file: str | Path | None = None,
) -> None:
    """Run all result plots for the workflow."""
    import ogstools as ot

    out_dir = Path(out_dir).resolve()
    ot.plot.setup.show_Region_bounds = False

    if pvd_file is None:
        pvd_file = find_any_pvd_file(out_dir, preferred_name=PVD_FILENAME)
    if pvd_file is None:
        raise FileNotFoundError(f"No PVD result file found under {out_dir}.")
    if initial_mesh_file is None:
        initial_mesh_file = find_mesh_file(out_dir)

    if initial_mesh_file is not None:
        save_initial_material_ids_png(out_dir=out_dir, mesh_file=initial_mesh_file)
    else:
        print("Skipping material_ids_initial.png because no initial mesh file was found.")

    ms = ot.MeshSeries(str(pvd_file))
    mesh = ms[-1]

    save_scalar_png_safe(
        mesh,
        scalar_name="pressure",
        title="Pressure At Final Timestep",
        png_path=out_dir / "pressure.png",
        opacity=0.2,
        clip=True,
        clip_normal=(0, 1, 0),
    )
    save_pressure_profile_with_velocity(
        mesh,
        png_path=out_dir / "pressure_profile.png",
    )
    save_velocity_contourf_png(
        mesh,
        png_path=out_dir / "fracture_velocity_contourf.png",
    )
    save_pressure_observation_png(
        ms,
        point=OBSERVATION_POINT,
        png_path=out_dir / PRESSURE_OBSERVATION_FIGURE,
    )

    print("Postprocessing completed!")
