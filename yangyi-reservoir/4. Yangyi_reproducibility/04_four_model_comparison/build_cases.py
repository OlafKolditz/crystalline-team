#!/usr/bin/env python3
"""Build the requested best-fit DFNM, DFNM-LRZ, and DFN cases."""

from __future__ import annotations

import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyvista as pv


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
JOINT007 = (
    REPO
    / "case1_k9_2e-11_k10_4e-13_noMatrix"
    / "random_wellhead_calibration"
    / "joint_downhole_head_calibration"
    / "head_refinement"
    / "projects"
    / "joint_007.prj"
)
FULL_BASE_DIR = REPO / "case1_k9_2e-11_k10_4e-13"
FULL_BASE_PROJECT = FULL_BASE_DIR / "c1_k9_2e-11_k10_4e-13_200d.prj"
NOMATRIX_BASE_DIR = REPO / "case1_k9_2e-11_k10_4e-13_noMatrix"
NOMATRIX_BASE_PROJECT = NOMATRIX_BASE_DIR / "c1_k9_2e-11_k10_4e-13_200d.prj"
DFN_BASE_DIR = REPO / "test-2D" / "case1_3D_initial_pressure"
DFN_BASE_PROJECT = DFN_BASE_DIR / "c1_3D_initial_pressure_200d.prj"
GENERATED_MESH_DIR = REPO / "Mesh" / "_out_yangyi_dfn_split"


MODELS = {
    "DFNM": {
        "base": FULL_BASE_PROJECT,
        "input_base": FULL_BASE_DIR,
        "prefix": "bestfit_DFNM_200d",
        "bulk_source": GENERATED_MESH_DIR / "yangyi_2d3d_fractures_matrix.vtu",
        "bulk_name": "reservoir_without_lrz.vtu",
    },
    "DFNM-LRZ": {
        "base": FULL_BASE_PROJECT,
        "input_base": FULL_BASE_DIR,
        "prefix": "bestfit_DFNM_LRZ_200d",
        "bulk_source": FULL_BASE_DIR / "input_mesh" / "reservoir_with_lowres.vtu",
        "bulk_name": "reservoir_with_lowres.vtu",
    },
    "DFN-LRZ": {
        "base": NOMATRIX_BASE_PROJECT,
        "input_base": NOMATRIX_BASE_DIR,
        "prefix": "bestfit_DFNM_LRZ_200d",
        "bulk_source": NOMATRIX_BASE_DIR / "input_mesh" / "reservoir_with_lowres.vtu",
        "bulk_name": "reservoir_with_lowres.vtu",
    },
    "DFN": {
        "base": DFN_BASE_PROJECT,
        "prefix": "bestfit_DFN_200d",
        "bulk_source": DFN_BASE_DIR / "yangyi_2d_fractures.vtu",
        "bulk_name": "yangyi_2d_fractures.vtu",
    },
}


def permeability_element(medium: ET.Element) -> ET.Element:
    for prop in medium.findall("./properties/property"):
        if prop.findtext("name") == "permeability":
            value = prop.find("value")
            if value is not None:
                return value
    raise RuntimeError(f"Medium {medium.attrib.get('id')} has no permeability value")


def permeability_map(project: Path) -> dict[int, float]:
    root = ET.parse(project).getroot()
    values = {}
    for medium in root.findall("./media/medium"):
        values[int(medium.attrib["id"])] = float(permeability_element(medium).text)
    return values


def copy_full_inputs(
    case_dir: Path, input_base: Path, bulk_source: Path, bulk_name: str
) -> None:
    input_dir = case_dir / "input_mesh"
    source_dir = case_dir / "source_terms"
    monitoring_dir = case_dir / "monitoring"
    input_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bulk_source, input_dir / bulk_name)
    for path in (input_base / "input_mesh").glob("yangyi_face_*.vtu"):
        shutil.copy2(path, input_dir / path.name)
    for path in (input_base / "source_terms").glob("*.vtu"):
        shutil.copy2(path, source_dir / path.name)
    monitor = (
        input_base
        / "monitoring"
        / "recommended_monitoring_bulk_nodes_all7_ZK501_bedrock.vtu"
    )
    shutil.copy2(monitor, monitoring_dir / monitor.name)


def copy_dfn_inputs(case_dir: Path) -> None:
    for name in ("yangyi_2d_fractures.vtu", "ZK403_2d_source.vtu", "ZK208_2d_source.vtu"):
        shutil.copy2(DFN_BASE_DIR / name, case_dir / name)


def build_project(model: str, settings: dict[str, object], calibrated: dict[int, float]) -> list[dict[str, object]]:
    case_dir = HERE / model
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "output").mkdir(exist_ok=True)
    tree = ET.parse(Path(settings["base"]))
    root = tree.getroot()

    if model == "DFN":
        copy_dfn_inputs(case_dir)
    else:
        copy_full_inputs(
            case_dir,
            Path(settings["input_base"]),
            Path(settings["bulk_source"]),
            str(settings["bulk_name"]),
        )
        first_mesh = root.find("./meshes/mesh")
        if first_mesh is None:
            raise RuntimeError("Full base project has no bulk mesh")
        first_mesh.text = f"input_mesh/{settings['bulk_name']}"

    records = []
    for medium in root.findall("./media/medium"):
        material_id = int(medium.attrib["id"])
        element = permeability_element(medium)
        previous = float(element.text)
        if material_id in calibrated:
            element.text = f"{calibrated[material_id]:.12e}"
            status = "transferred_from_joint_007"
        else:
            status = "retained_from_base_not_in_joint_007"
        records.append(
            {
                "model": model,
                "material_id": material_id,
                "base_permeability_m2": previous,
                "final_permeability_m2": float(element.text),
                "status": status,
            }
        )

    prefix = root.find("./time_loop/output/prefix")
    if prefix is None:
        raise RuntimeError(f"{model} base project has no output prefix")
    prefix.text = str(settings["prefix"])
    ET.indent(tree, space="  ")
    project = case_dir / f"{settings['prefix']}.prj"
    tree.write(project, encoding="UTF-8", xml_declaration=True)
    return records


def audit_source(case_dir: Path, bulk_path: Path, source_paths: list[Path]) -> list[dict[str, object]]:
    bulk = pv.read(bulk_path)
    rows = []
    for path in source_paths:
        source = pv.read(path)
        bulk_id = int(np.asarray(source.point_data["bulk_node_ids"]).reshape(-1)[0])
        if not 0 <= bulk_id < bulk.n_points:
            raise RuntimeError(f"Invalid bulk node {bulk_id} in {path}")
        distance = float(np.linalg.norm(bulk.points[bulk_id] - source.points[0]))
        if distance > 1e-3:
            raise RuntimeError(f"Source mapping mismatch in {path}: {distance:g} m")
        rows.append(
            {
                "model": case_dir.name,
                "source_mesh": path.name,
                "bulk_node_id": bulk_id,
                "coordinate_mismatch_m": distance,
            }
        )
    return rows


def main() -> None:
    calibrated = permeability_map(JOINT007)
    transfer_rows = []
    for model, settings in MODELS.items():
        transfer_rows.extend(build_project(model, settings, calibrated))

    audit_rows = []
    for model, settings in MODELS.items():
        case_dir = HERE / model
        if model == "DFN":
            bulk = case_dir / "yangyi_2d_fractures.vtu"
            sources = [case_dir / "ZK403_2d_source.vtu", case_dir / "ZK208_2d_source.vtu"]
        else:
            bulk = case_dir / "input_mesh" / str(settings["bulk_name"])
            sources = sorted((case_dir / "source_terms").glob("*.vtu"))
        audit_rows.extend(audit_source(case_dir, bulk, sources))

    with (HERE / "permeability_transfer.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(transfer_rows[0]))
        writer.writeheader()
        writer.writerows(transfer_rows)
    with (HERE / "source_mapping_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    shutil.copy2(JOINT007, HERE / "joint_007_permeability_source.prj")
    monitoring_dir = HERE / "monitoring_data"
    monitoring_dir.mkdir(exist_ok=True)
    shutil.copy2(
        FULL_BASE_DIR / "monitoring" / "yangyi_after_20190113_for_ogs.xlsx",
        monitoring_dir / "yangyi_after_20190113_for_ogs.xlsx",
    )
    print("Built DFNM, DFNM-LRZ, DFN, and DFN-LRZ projects.")
    print("Transferred permeabilities:", ", ".join(f"M{k}={v:.12e}" for k, v in calibrated.items()))
    print("All source-node mappings passed the coordinate audit.")


if __name__ == "__main__":
    main()
