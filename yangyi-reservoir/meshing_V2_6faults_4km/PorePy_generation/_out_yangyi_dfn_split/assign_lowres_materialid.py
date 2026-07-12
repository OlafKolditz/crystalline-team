import pyvista as pv
import numpy as np
import sys
from pathlib import Path


def main():
    # ===== 输入文件 =====
    bulk_mesh_file = "yangyi_2d3d_fractures_matrix.vtu"                   # 你的储层主网格
    closed_zone_file = "3.5ohm_closed_components.vtp"  # 你刚提取出的封闭低阻带
    output_file = "reservoir_with_lowres.vtu"          # 输出新网格

    # ===== 参数 =====
    old_id = 0   # 只修改原本是 0 的单元
    new_id = 12   # 映射进去后赋成 12

    # ===== 读取网格 =====
    print(f"读取主网格: {bulk_mesh_file}")
    bulk = pv.read(bulk_mesh_file)

    print(f"读取低阻带封闭面: {closed_zone_file}")
    zone = pv.read(closed_zone_file)

    # ===== 基本检查 =====
    if bulk.n_cells == 0:
        raise RuntimeError("主网格没有单元，请检查 reservoir.vtu")

    if zone.n_cells == 0:
        raise RuntimeError("低阻带 vtp 没有面单元，请检查 3.5ohm_closed_components.vtp")

    if "MaterialIDs" not in bulk.cell_data:
        raise RuntimeError("主网格中没有 cell_data['MaterialIDs']，请先确认主网格里已有材料编号。")

    mat_ids = np.array(bulk.cell_data["MaterialIDs"]).copy()

    if len(mat_ids) != bulk.n_cells:
        raise RuntimeError("MaterialIDs 数组长度和主网格单元数不一致。")

    print(f"主网格单元数: {bulk.n_cells}")
    print(f"低阻带表面单元数: {zone.n_cells}")

    # ===== 提取单元中心 =====
    print("计算主网格单元中心...")
    centers = bulk.cell_centers()

    # ===== 判断哪些单元中心位于封闭低阻带内部 =====
    # check_surface=True 会检查 zone 是否可用于 enclosed test
    print("判断哪些单元中心落在低阻带内部...")
    selected = centers.select_enclosed_points(
        zone,
        tolerance=1e-8,
        check_surface=True
    )

    if "SelectedPoints" not in selected.point_data:
        raise RuntimeError("select_enclosed_points 没有返回 SelectedPoints。")

    inside = np.array(selected.point_data["SelectedPoints"]).astype(bool)

    # ===== 只修改原本 MaterialIDs == 0 的单元 =====
    candidate = (mat_ids == old_id)
    to_change = inside & candidate

    n_inside = int(np.sum(inside))
    n_candidate = int(np.sum(candidate))
    n_change = int(np.sum(to_change))

    print(f"落在低阻带内部的单元数: {n_inside}")
    print(f"当前 MaterialIDs == {old_id} 的单元数: {n_candidate}")
    print(f"最终要改成 {new_id} 的单元数: {n_change}")

    # ===== 修改 MaterialIDs =====
    mat_ids[to_change] = new_id
    bulk.cell_data["MaterialIDs"] = mat_ids

    # ===== 保存 =====
    print(f"保存输出网格: {output_file}")
    bulk.save(output_file)

    # ===== 统计输出 =====
    unique_ids, counts = np.unique(mat_ids, return_counts=True)
    print("\n输出网格中的 MaterialIDs 统计:")
    for mid, cnt in zip(unique_ids, counts):
        print(f"  ID = {mid}: {cnt} cells")

    print("\n完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)
