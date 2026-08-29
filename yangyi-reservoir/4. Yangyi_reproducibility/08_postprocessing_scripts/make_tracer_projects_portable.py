#!/usr/bin/env python3
"""Create documented path-only copies of retained tracer projects."""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / '05_tracer_calibration_sweeps' / 'runs'
for source in sorted(RUNS.glob('*/*.prj')):
    if source.stem.endswith('_portable'):
        continue
    tree = ET.parse(source)
    root = tree.getroot()
    for node in root.findall('./meshes/mesh'):
        if node.text:
            local = Path('input_mesh') / Path(node.text.strip()).name
            if not (source.parent / local).is_file():
                raise FileNotFoundError(f'{source}: {local}')
            node.text = local.as_posix()
    target = source.with_name(source.stem + '_portable.prj')
    if target.exists():
        raise FileExistsError(target)
    ET.indent(tree, space='  ')
    tree.write(target, encoding='UTF-8', xml_declaration=True)
    print(target.relative_to(ROOT))
