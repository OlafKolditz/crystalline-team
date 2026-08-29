#!/usr/bin/env python3
"""Render final pressure-change and Darcy-velocity fields for the four cases."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("MESA_SHADER_CACHE_DISABLE", "true")

import numpy as np
import pyvista as pv


HERE = Path(__file__).resolve().parent
OUT = HERE / "paper_draft" / "figures"
CASES = {
    "DFNM": HERE / "DFNM" / "output" / "bestfit_DFNM_200d.pvd",
    "DFNM-LRZ": HERE / "DFNM-LRZ" / "output" / "bestfit_DFNM_LRZ_200d.pvd",
    "DFN": HERE / "DFN" / "output" / "bestfit_DFN_200d.pvd",
    "DFN-LRZ": HERE / "DFN-LRZ" / "output" / "bestfit_DFNM_LRZ_200d.pvd",
}
INPUT_MESHES = {
    "DFNM": HERE / "DFNM" / "input_mesh" / "reservoir_without_lrz.vtu",
    "DFNM-LRZ": HERE / "DFNM-LRZ" / "input_mesh" / "reservoir_with_lowres.vtu",
    "DFN": HERE / "DFN" / "yangyi_2d_fractures.vtu",
    "DFN-LRZ": HERE / "DFN-LRZ" / "input_mesh" / "reservoir_with_lowres.vtu",
}


def endpoints(pvd: Path) -> tuple[Path, Path, float]:
    datasets = ET.parse(pvd).getroot().findall(".//DataSet")
    return (
        pvd.parent / datasets[0].attrib["file"],
        pvd.parent / datasets[-1].attrib["file"],
        float(datasets[-1].attrib["timestep"]) / 86_400.0,
    )


def display_parts(mesh: pv.UnstructuredGrid, model: str) -> list[tuple[pv.DataSet, float]]:
    """Expose fracture/LRZ surfaces and a central matrix section."""
    ids = np.asarray(mesh.cell_data["MaterialIDs"])
    parts: list[tuple[pv.DataSet, float]] = []
    if np.any(ids == 0):
        matrix = mesh.extract_cells(ids == 0)
        parts.append((matrix.slice(normal="y", origin=mesh.center), 1.0))
    nonmatrix = mesh.extract_cells(ids != 0)
    if nonmatrix.n_cells:
        parts.append((nonmatrix, 0.85))
    return parts


def render(field: str, output: Path) -> None:
    plotter = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1800, 1250))
    plotter.set_background("white")
    for index, (model, pvd) in enumerate(CASES.items()):
        plotter.subplot(index // 2, index % 2)
        first_path, final_path, day = endpoints(pvd)
        initial = pv.read(first_path)
        final = pv.read(final_path)
        material_mesh = pv.read(INPUT_MESHES[model])
        final.cell_data["MaterialIDs"] = np.asarray(
            material_mesh.cell_data["MaterialIDs"]
        )
        if field == "pressure_change_MPa":
            values = (
                np.asarray(final.point_data["pressure"])
                - np.asarray(initial.point_data["pressure"])
            ) / 1.0e6
            final.point_data[field] = values
            clim = (-1.0, 2.0)
            cmap = "coolwarm"
            title = f"{model} (day {day:g}); raw range {values.min():.2g} to {values.max():.2g} MPa"
            scalar_title = "delta p [MPa] (clipped)"
        else:
            velocity = np.linalg.norm(
                np.asarray(final.point_data["DarcyVelocity"]), axis=1
            )
            values = np.log10(np.maximum(velocity, 1.0e-16))
            final.point_data[field] = values
            clim = (-12.0, -3.0)
            cmap = "viridis"
            title = f"{model} (day {day:g}); max |q|={velocity.max():.2e} m/s"
            scalar_title = "log10 |q| [m/s]"
        for part_index, (shown, opacity) in enumerate(display_parts(final, model)):
            plotter.add_mesh(
                shown,
                scalars=field,
                cmap=cmap,
                clim=clim,
                opacity=opacity,
                show_edges=False,
                show_scalar_bar=part_index == 0,
                scalar_bar_args={
                    "title": scalar_title,
                    "title_font_size": 13,
                    "label_font_size": 11,
                    "color": "black",
                },
            )
        plotter.add_text(title, position="upper_left", font_size=11, color="black")
        plotter.camera_position = "iso"
        plotter.camera.zoom(1.15)
        plotter.show_axes()
    plotter.show(screenshot=output, auto_close=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    render("pressure_change_MPa", OUT / "final_pressure_change_fields.png")
    render("log10_Darcy_velocity", OUT / "final_darcy_velocity_fields.png")
    print(f"Created final-field figures in {OUT}")


if __name__ == "__main__":
    main()
