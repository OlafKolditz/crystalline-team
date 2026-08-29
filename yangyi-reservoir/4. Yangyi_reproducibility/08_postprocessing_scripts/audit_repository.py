#!/usr/bin/env python3
"""Regenerate the repository manifest and OGS mesh-reference audit."""
from pathlib import Path
import csv
import hashlib
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

def category_root(path: Path) -> Path:
    rel = path.relative_to(ROOT)
    return ROOT / rel.parts[0]

rows = []
for project in sorted(ROOT.rglob('*.prj')):
    root = ET.parse(project).getroot()
    refs = [x.text.strip() for x in root.findall('./meshes/mesh') if x.text]
    for ref in refs:
        raw = Path(ref)
        candidates = [project.parent / raw]
        parent = project.parent
        cat = category_root(project)
        while parent != cat and cat in parent.parents:
            parent = parent.parent
            candidates.append(parent / raw)
        found = next((p for p in candidates if p.is_file()), None)
        if raw.is_absolute():
            status = 'absolute_reference_present' if raw.is_file() else 'absolute_reference_missing'
        elif found == project.parent / raw:
            status = 'found_project_directory'
        elif found:
            status = 'found_ancestor_working_directory'
        else:
            status = 'missing'
        rows.append({
            'dependency_type': 'project_mesh',
            'file': project.relative_to(ROOT).as_posix(),
            'reference': ref,
            'status': status,
            'resolved_path': (found.relative_to(ROOT).as_posix() if found and ROOT in found.parents else str(found)) if found else '',
            'notes': 'Nested calibration projects must be launched from their category root.' if status == 'found_ancestor_working_directory' else '',
        })

manual = [
    ('script_external', '03_DFN_LRZ_calibration/random_wellhead_calibration/run_random_calibration.py', '../case1_k9_2e-11_k10_4e-13/compare_observed_wellhead_pressure.py', 'missing_external', 'Required to regenerate/score the calibration; retained .prj files remain runnable.'),
    ('script_external', '03_DFN_LRZ_calibration/random_wellhead_calibration/joint_downhole_head_calibration/run_joint_calibration.py', 'sibling case1_k9_2e-11_k10_4e-13 pressure helper', 'missing_external', 'Required for pressure scoring.'),
    ('script_external', '04_four_model_comparison/build_cases.py', 'original full/no-matrix/Surface-DFN base directories', 'missing_external', 'Only needed to regenerate final comparison projects; final projects are retained.'),
    ('script_portable', '04_four_model_comparison/plot_well_pressure_comparisons_portable.py', 'DFNM-LRZ/source_terms and monitoring_data workbook', 'resolved_portable_copy', 'Path-only copy; original preserved.'),
    ('script_portable', '04_four_model_comparison/plot_monitoring_head_comparisons_portable.py', 'monitoring_data mapping CSVs and workbook', 'resolved_portable_copy', 'Path-only copy; original preserved.'),
    ('project_portable', '05_tracer_calibration_sweeps/runs/*/*_portable.prj', 'per-run input_mesh', 'resolved_portable_copy', '20 path-only copies; absolute-path originals preserved.'),
    ('script_external', '05_tracer_calibration_sweeps/run_joint_sweep.py', '../case6_4pipes_deep_reservoir_feedback/run_deep_feedback.py', 'missing_external', 'Required to regenerate feedback curves and score the 20-case sweep.'),
    ('script_external', '05_tracer_calibration_sweeps/run_joint_sweep.py', '../case6_4pipes_temperature_constrained_flow_sweep/run_flow_fraction_sweep.py', 'missing_external', 'Required to regenerate allocation curves.'),
    ('script_external', '05_tracer_calibration_sweeps/run_joint_sweep.py', '../case6_4pipes_deep_release_time_sweep/run_tau_sweep.py', 'missing_external', 'Required helper; individual retained projects run without it.'),
    ('script_external', '06_thermal_consistency/run_joint_refinement.py', '../case3_split_storage_heat_exchange_4y/dispersion_runs/.../*.prj', 'missing_external', 'Parent/template required to regenerate projects; final nine projects are retained.'),
    ('script_external', '06_thermal_consistency/run_joint_refinement.py', '../case3_split_storage_heat_exchange_4y/Power output and PT.xls', 'missing_external', 'Required to score ZK203/ZK208 temperatures.'),
    ('script_external', '06_thermal_consistency/run_joint_refinement.py', '../case3_split_storage_heat_exchange_4y/.python_deps', 'missing_external', 'Original local package environment; install declared imports independently.'),
    ('software', 'multiple scripts', '/home/zhai/ogs-env/bin/ogs and /home/zhai/ogs-env/bin/python', 'environment_specific', 'Set OGS_BIN where supported or invoke retained projects with local executables.'),
    ('script_context', '08_postprocessing_scripts/*', 'case-relative inputs', 'discoverability_copy_only', 'Use case-local original scripts; central copies intentionally index source code.'),
]
for dep_type, file, ref, status, notes in manual:
    rows.append({'dependency_type': dep_type, 'file': file, 'reference': ref,
                 'status': status, 'resolved_path': '', 'notes': notes})

with (ROOT / 'DEPENDENCY_AUDIT.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=['dependency_type','file','reference','status','resolved_path','notes'])
    writer.writeheader(); writer.writerows(rows)

manifest = []
for path in sorted(ROOT.rglob('*')):
    if not path.is_file() or path.name == 'FILE_MANIFEST.csv':
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest})
with (ROOT / 'FILE_MANIFEST.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=['path','bytes','sha256'])
    writer.writeheader(); writer.writerows(manifest)

missing_projects = sum(r['status'] == 'missing' for r in rows if r['dependency_type'] == 'project_mesh')
print(f'projects={sum(1 for _ in ROOT.rglob("*.prj"))} project_mesh_refs={sum(r["dependency_type"] == "project_mesh" for r in rows)} missing_project_mesh_refs={missing_projects} manifest_files={len(manifest)}')
