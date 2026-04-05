# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python (.venv)
#     language: python
#     name: venv
# ---

# %% [raw]
# +++
# title = "OpenGeoSys Workflow for STIMTEC DFN model"
# date = "2025-04-28"
# author = "Mostafa Mollaali"
# web_subsection = "reactive-transport"
# +++

# %%
import json
import os
import platform
import shutil
from pathlib import Path

import numpy as np
import ogstools as ot
import pyvista as pv
import matplotlib.pyplot as plt
from IPython.display import Image as IPyImage
from IPython.display import Markdown, display
from numpy.random import default_rng
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pyvista as pv
import vtk

from pathlib import Path
import json
import numpy as np



# %%
ot.plot.setup.show_Region_bounds = False

out_dir = Path(os.environ.get("OGS_TESTRUNNER_OUT_DIR", "_out")).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

OBSERVATION_POINT = np.array([4.59509e6, 5.6442e6, 268.933], dtype=float)
PRESSURE_OBSERVATION_FIGURE = "pressure_observation_point.png"


# %% [markdown]
# ## STIMTEC DFN Workflow
#
# This script reproduces the mixed-dimensional STIMTEC liquid-flow example used in
# this repository. It assumes the bulk mesh, borehole seed mesh, and OGS project
# file already exist next to the script:
#
# - ``mixed_dimensional_all.vtu``
# - ``BH10.vtu``
# - ``STIMTEC_DFN.prj``
#


# %% [markdown]
# # DFN generating using Porepy

# %% [markdown]
# ### Input data

# %% [markdown]
# The current project file defines a single-phase liquid-flow process with:
#
# - gravity: ``(0, 0, -9.8) m/s^2``
# - viscosity: ``1e-3 Pa s``
# - density: linearized around ``1000 kg/m^3`` with slope ``4.5e-7``
# - matrix porosity: ``0.005``
# - fracture porosity: ``0.005``
# - matrix permeability: ``1e-8 m^2``
# - fracture permeabilities: ``kappa_frac1`` to ``kappa_frac4`` from
#   ``STIMTEC_DFN.prj``
# - hydrostatic initial pressure ``p0``
# - borehole loading on ``BH10`` ramping to ``1.88e6 Pa`` between
#   ``1000 s`` and ``2000 s``

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

# Borehole length
dx = 27.6546
dy = 45.6163
dz = 15.7221
L_borehole = np.sqrt(dx**2 + dy**2 + dz**2)
print("L_borehole",L_borehole)
def q_in(t):
    return np.where(t < 1794.21, 4.99798e-08,
           np.where(t < 1909.81, 9.31177e-06,
           np.where(t < 2079.81, 1.90107e-05,
           np.where(t < 2160.01, 3.41504e-05,
           np.where(t < 2330.81, 6.7659e-05,
           np.where(t < 2430.41, 1.01353e-04,
           np.where(t < 2517.81, 1.33182e-04,
           np.where(t < 2669.81, 2.52185e-07,
                    0.0))))))))

t = np.linspace(0, 3000, 1000)

y_total = q_in(t)
y_per_length = y_total / L_borehole

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

# --- Left: original ---
axes[0].step(t, y_total, where="post", linewidth=2.5)
axes[0].set_title(r"Total flow rate $q_{in}(t)$")
axes[0].set_xlabel(r"Time, $t$ / s")
axes[0].set_ylabel(r"$q_{in}(t)$ / m$^3$ s$^{-1}$")

# --- Right: per length ---
axes[1].step(t, y_per_length, where="post", linewidth=2.5)
axes[1].set_title(r"Flow per unit length $q_{in}(t)/L_b$")
axes[1].set_xlabel(r"Time, $t$ / s")
axes[1].set_ylabel(r"$q_{in}(t)/L_b$ / m$^2$ s$^{-1}$")

for ax in axes:
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.set_xlim(0, 3000)
    ax.margins(x=0.01, y=0.08)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Remove duplicated nodes in the mesh using PyVista

# %%
from pathlib import Path
import pyvista as pv
import shutil

# Paths
orig_dir = Path.cwd().resolve()
src_file = orig_dir / "mixed_dimensional_all.vtu"
src_bh10 = orig_dir / "BH10.vtu"

dst_dir = out_dir
dst_file = dst_dir / "mixed_dimensional_all.vtu"
dst_bh10 = dst_dir / "BH10.vtu"

# Ensure output directory exists
dst_dir.mkdir(parents=True, exist_ok=True)

# Copy meshes into the output directory, overwriting prior files if needed.
for src, dst in ((src_file, dst_file), (src_bh10, dst_bh10)):
    if not src.exists():
        raise FileNotFoundError(f"Required mesh file not found: {src}")
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.copy2(src, dst)

# --- Work with mesh ---
mesh_c = pv.read(dst_file)

clean_mesh = mesh_c.clean(tolerance=1e-12)
clean_mesh.save(dst_file)

print(f"Original points: {mesh_c.n_points}, Cleaned points: {clean_mesh.n_points}")
print(f"Mesh workspace: {dst_dir.resolve()}")

# %%


# (Optional) make VTK happier in headless contexts
pv.OFF_SCREEN = True


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
):
    png_path = Path(png_path)

    # ---- Ensure mesh is a PyVista dataset ----
    # ot.MeshSeries often returns something already compatible, but let's be safe:
    if not isinstance(mesh, pv.DataSet):
        try:
            mesh = pv.wrap(mesh)
        except Exception as e:
            raise TypeError(f"mesh is not a PyVista dataset and cannot be wrapped. type={type(mesh)}") from e

    # ---- Validate scalar exists (point or cell) ----
    has_point = scalar_name in mesh.point_data
    has_cell  = scalar_name in mesh.cell_data
    if not (has_point or has_cell):
        raise KeyError(
            f"Scalar '{scalar_name}' not found.\n"
            f"point_data keys: {list(mesh.point_data.keys())}\n"
            f"cell_data keys : {list(mesh.cell_data.keys())}"
        )

    if clip:
        mesh = mesh.clip(normal=clip_normal)

    # If it’s cell data, tell pyvista explicitly
    preference = "cell" if has_cell and not has_point else "point"

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1000))

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

    plotter.add_mesh(
        mesh,
        **add_mesh_kwargs,
    )

    plotter.add_mesh(mesh.outline(), color="black", line_width=2)
    plotter.show_axes()
    plotter.enable_parallel_projection()
    plotter.view_isometric()

    png_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(png_path))
    plotter.close()

    print("Saved:", png_path)
    display(Markdown(f"**{title}**"))
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


def save_pressure_profile_with_velocity(mesh, png_path: str | Path):
    png_path = Path(png_path)

    mesh = _mesh_with_point_field(mesh, "pressure")
    mesh, vector_name = _resolve_velocity_field(mesh)
    arrows = _build_velocity_glyphs(mesh, vector_name)

    scalar_bar_args = dict(base_scalar_bar_args)
    scalar_bar_args["title"] = "Pressure [Pa]"

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1000))
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
    display(Markdown("**Pressure Profile With Darcy Velocity**"))
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


def save_ogstools_contourf_png(mesh, variable, png_path: str | Path, title: str, **kwargs):
    png_path = Path(png_path)
    fig = ot.plot.contourf(mesh, variable, **kwargs)
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", png_path)
    display(Markdown(f"**{title}**"))
    display(IPyImage(filename=str(png_path)))


def save_velocity_contourf_png(mesh, png_path: str | Path):
    fracture_mesh = extract_fracture_mesh(mesh)
    if "v" in fracture_mesh.point_data:
        if "velocity" not in fracture_mesh.point_data:
            fracture_mesh.point_data["velocity"] = fracture_mesh.point_data["v"]
        if "darcy_velocity" not in fracture_mesh.point_data:
            fracture_mesh.point_data["darcy_velocity"] = fracture_mesh.point_data["v"]

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


def save_pressure_observation_png(
    mesh_series: ot.MeshSeries,
    point: np.ndarray,
    png_path: str | Path,
):
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
    display(Markdown("**Pressure At Observation Point**"))
    display(IPyImage(filename=str(png_path)))


def save_material_ids_png(mesh, png_path: str | Path):
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


initial_mesh = pv.read(dst_file)
save_material_ids_png(initial_mesh, out_dir / "material_ids_initial.png")

# %% [markdown]
# ## Extract boundary conditions

# %%
xmin, xmax, ymin, ymax, zmin, zmax = clean_mesh.bounds
Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin
domain_size = max(Lx, Ly, Lz)

boundary_tol_fraction = 1e-10
tol = boundary_tol_fraction * domain_size

# Target planes (slightly inside the domain)
x_left   = xmin + tol
x_right  = xmax - tol
y_front  = ymin + tol
y_back   = ymax - tol
z_bottom = zmin + tol
z_top    = zmax - tol

print("Bounds:", clean_mesh.bounds)
print("tol =", tol)


# %%
ot.cli().NodeReordering(
    i=str(dst_file), o=str(dst_file),   m=0,
    no_volume_check=True,
)

# %%
outer_boundary = dst_dir / "outer_boundary.vtu"
ot.cli().ExtractBoundary(i=str(dst_file), o=str(outer_boundary))
# x = xmin (LEFT)
ot.cli().removeMeshElements(
    i=str(outer_boundary), o=str(dst_dir / "bc_xmin.vtu"),
    **{"x-max": x_left, "invert": True}
)

# x = xmax (RIGHT)
ot.cli().removeMeshElements(
    i=str(outer_boundary), o=str(dst_dir / "bc_xmax.vtu"),
    **{"x-min": x_right, "invert": True}
)

# y = ymin (FRONT)
ot.cli().removeMeshElements(
    i=str(outer_boundary), o=str(dst_dir / "bc_ymin.vtu"),
    **{"y-max": y_front, "invert": True}
)

# y = ymax (BACK)
ot.cli().removeMeshElements(
    i=str(outer_boundary), o=str(dst_dir / "bc_ymax.vtu"),
    **{"y-min": y_back, "invert": True}
)

# z = zmin (BOTTOM)
ot.cli().removeMeshElements(
    i=str(outer_boundary), o=str(dst_dir / "bc_zmin.vtu"),
    **{"z-max": z_bottom, "invert": True}
)

# z = zmax (TOP)
ot.cli().removeMeshElements(
    i=str(outer_boundary), o=str(dst_dir / "bc_zmax.vtu"),
    **{"z-min": z_top, "invert": True}
)



# %%
# --- your original seeds (keep format)
seed_meshes = [
    "bc_xmin.vtu",
    "bc_xmax.vtu",
    "bc_ymin.vtu",
    "bc_ymax.vtu",
    "bc_zmin.vtu",
    "bc_zmax.vtu",
    "BH10.vtu",
]

# --- add borehole seeds (now copied into out_dir root)
# seed_meshes += [f.name for f in well_seeds]

cwd_before_identify = Path.cwd()
os.chdir(dst_dir)
try:
    ot.cli().identifySubdomains(
        "-m", "mixed_dimensional_all.vtu",
        # "-s", str(1e-10),
        "--",
        *seed_meshes,
    )
finally:
    os.chdir(cwd_before_identify)

# %% [markdown]
# ## Project file and run OpenGeosys

# %%
project_file = orig_dir / "STIMTEC_DFN.prj"

project = ot.Project(
    input_file=project_file, output_file=dst_dir / "STIMTEC_DFN_final.prj"
)
project.write_input()
project.run_model(args=f"-o {dst_dir} -m {dst_dir}", logfile=dst_dir / "run.log")


# %% [markdown]
# ## Post-processing

# %%
ms = ot.MeshSeries(str(out_dir / "Stimtec_DFN.pvd"))
mesh = ms[-1]

# Final pressure views
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
