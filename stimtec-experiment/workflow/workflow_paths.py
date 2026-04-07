"""File discovery helpers for the STIMTEC workflow."""

from pathlib import Path
import xml.etree.ElementTree as ET


IGNORED_SEARCH_DIR_NAMES = {"__pycache__"}
DEPRIORITIZED_SEARCH_DIR_NAMES = {"_out"}

MESH_CANDIDATE_NAMES = ("mixed_dimensional_all.vtu", "mesh.vtu")
BOREHOLE_SEED_CANDIDATE_NAMES = ("BH10.vtu",)
PROJECT_CANDIDATE_NAMES = ("STIMTEC_DFN.prj", "model01.prj")
PVD_CANDIDATE_NAMES = ("Stimtec_DFN.pvd",)


def _filtered_matches(search_dir: Path, matches: list[Path]) -> list[Path]:
    filtered: list[Path] = []
    for path in matches:
        relative_parts = set(path.relative_to(search_dir).parts[:-1])
        if relative_parts & IGNORED_SEARCH_DIR_NAMES:
            continue
        filtered.append(path)
    return filtered


def _sorted_matches(search_dir: Path, matches: list[Path]) -> list[Path]:
    filtered_matches = _filtered_matches(search_dir, matches)

    def sort_key(path: Path) -> tuple[int, int, str]:
        relative_parts = path.relative_to(search_dir).parts[:-1]
        deprioritized_count = sum(part in DEPRIORITIZED_SEARCH_DIR_NAMES for part in relative_parts)
        return deprioritized_count, len(relative_parts), str(path)

    return sorted(filtered_matches, key=sort_key)


def find_named_file(search_dir: Path, candidate_names: tuple[str, ...]) -> Path | None:
    """Find the first matching file in a directory tree."""
    search_dir = Path(search_dir).resolve()

    for name in candidate_names:
        direct_match = search_dir / name
        if direct_match.exists():
            return direct_match

    for name in candidate_names:
        matches = _sorted_matches(search_dir, list(search_dir.rglob(name)))
        if matches:
            return matches[0]

    return None


def find_any_pvd_file(search_dir: Path, preferred_name: str | None = None) -> Path | None:
    """Find a PVD result file in the provided directory tree."""
    search_dir = Path(search_dir).resolve()

    if preferred_name:
        direct_match = search_dir / preferred_name
        if direct_match.exists():
            return direct_match

        matches = _sorted_matches(search_dir, list(search_dir.rglob(preferred_name)))
        if matches:
            return matches[0]

    named_match = find_named_file(search_dir, PVD_CANDIDATE_NAMES)
    if named_match is not None:
        return named_match

    matches = _sorted_matches(search_dir, list(search_dir.rglob("*.pvd")))
    if matches:
        return matches[0]

    return None


def find_mesh_file(search_dir: Path) -> Path | None:
    return find_named_file(search_dir, MESH_CANDIDATE_NAMES)


def find_borehole_seed_file(search_dir: Path) -> Path | None:
    return find_named_file(search_dir, BOREHOLE_SEED_CANDIDATE_NAMES)


def find_project_file(search_dir: Path) -> Path | None:
    return find_named_file(search_dir, PROJECT_CANDIDATE_NAMES)


def read_project_mesh_names(project_file: Path) -> list[str]:
    """Read mesh names referenced by an OGS project file."""
    root = ET.parse(project_file).getroot()
    return [element.text.strip() for element in root.findall(".//meshes/mesh") if element.text]


def read_project_output_prefix(project_file: Path) -> str | None:
    """Read the output prefix from an OGS project file."""
    root = ET.parse(project_file).getroot()
    prefix = root.find(".//time_loop/output/prefix")
    if prefix is None or prefix.text is None:
        return None
    return prefix.text.strip()
