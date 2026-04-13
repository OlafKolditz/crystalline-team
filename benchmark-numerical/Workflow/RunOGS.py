import argparse
import os
import sys
from pathlib import Path

import numpy as np
import ogstools as ot


def run_project(
    prj_file,
    output_prefix,
    out_dir,
    mesh_dir,
    ogs_path=None,
    order=1,
):
    prj_temp = f"{output_prefix}.prj"
    prj = ot.Project(
        input_file=prj_file,
        output_file=Path(out_dir, prj_temp),
        OMP_NUM_THREADS=4,
        OGS_ASM_THREADS=4,
    )

    xpath_var = './process_variables/process_variable[name="displacement"]/'
    prj.replace_text(
        order,
        xpath=xpath_var + "order",
    )

    prj.replace_text(
        output_prefix,
        xpath="./time_loop/output/prefix",
    )

    prj.write_input()

    pvd_file = Path(out_dir, f"{output_prefix}.pvd")
    print(f"Output PVD file: {pvd_file}")

    prj.run_model(
        logfile=Path(out_dir, "log.txt"),
        path=ogs_path,
        args=f"-o {out_dir} -m {mesh_dir}",
    )


def execute(prj_file, prefix, ogs_path, out_dir, mesh_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_project(
            prj_file,
            prefix,
            out_dir,
            mesh_dir,
            ogs_path=ogs_path,
        )
        print("\nOGS completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process mesh and output paths.")
    parser.add_argument(
        "-e",
        "--ogs_path",
        default=None,
        help="Path to the OGS executable file. If it not given, "
             "the default path in the environment is taken."
    )
    parser.add_argument(
        "-m",
        "--mesh",
        default=None,
        help="Path to the mesh file or directory. "
             "If it is not given, the default mesh path is ../Mesh"
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to the output directory. "
             " If it is not given, the default mesh path is ./_out"
    )

    args = parser.parse_args()
    ogs_path = None if args.ogs_path is None else Path(args.ogs_path)
    if ogs_path is not None:
        print(f"Path to ogs: {ogs_path}.")
    out_dir = Path("_out") if args.output is None else Path(args.output)
    print(f"Output path: {out_dir}.")
    mesh_dir = Path("..", "Mesh") if args.mesh is None else Path(args.output, args.mesh)
    print(f"Mesh path: {mesh_dir}.")

    prj_file = Path("..", "simulation_with_HM.prj")
    prefix = "simulation_with_HM"
    execute(prj_file, prefix, ogs_path, out_dir, mesh_dir)

    prj_file_H = Path("..", "simulation_with_H.prj")
    prefix_H = "simulation_with_H"
    execute(prj_file_H, prefix_H, ogs_path, out_dir, mesh_dir)
