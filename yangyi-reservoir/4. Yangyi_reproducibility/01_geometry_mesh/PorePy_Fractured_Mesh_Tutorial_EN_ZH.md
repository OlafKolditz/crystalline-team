---
title: "Generating a 3D Fractured Mesh with PorePy and Gmsh / 使用 PorePy 和 Gmsh 生成三维裂隙网格"
author: "Yangyi DFNM workflow / 阳易 DFNM 工作流程"
date: "August 2026 / 2026年8月"
lang: en-US
---

# Purpose / 教程目的

**English.** This tutorial describes how to convert a table of interpreted fault parameters into a conforming, mixed-dimensional mesh using PorePy and Gmsh. The final mesh contains a three-dimensional rock matrix, two-dimensional fracture surfaces, and one-dimensional fracture-intersection lines. It can be inspected in ParaView and converted to an OpenGeoSys-ready VTU mesh.

**中文。** 本教程介绍如何利用 PorePy 和 Gmsh，将解释得到的断层参数表转换为几何相容的混合维度网格。最终网格包含三维岩石基质、二维裂隙面以及一维裂隙交线，可在 ParaView 中检查，并可转换为适用于 OpenGeoSys 的 VTU 网格。

The complete workflow is / 完整流程为：

```text
Fault parameters in CSV / CSV断层参数
                 ↓
3D elliptical polygons / 三维椭圆断层面
                 ↓
PorePy fracture network / PorePy裂隙网络
                 ↓
Gmsh conforming simplex mesh / Gmsh相容单纯形网格
                 ↓
3D matrix + 2D fractures + 1D intersections
三维基质 + 二维裂隙 + 一维交线
                 ↓
VTU meshes for ParaView and OpenGeoSys
用于ParaView和OpenGeoSys的VTU网格
```

# 1. Mesh concept / 网格概念

**English.** PorePy represents a fractured reservoir as a mixed-dimensional system. The rock matrix is discretized with 3D tetrahedra, each fracture is discretized with 2D triangles, and the intersection between two fractures is represented by 1D line elements. A conforming mesh means that fracture triangles coincide with tetrahedral faces and intersection lines coincide with fracture-triangle edges.

**中文。** PorePy 将裂隙储层表示为混合维度系统。岩石基质采用三维四面体单元离散，裂隙采用二维三角形单元离散，两条裂隙之间的交线采用一维线单元表示。所谓相容网格，是指裂隙三角形与四面体面完全重合，并且裂隙交线与裂隙三角形的边完全重合。

| Dimension / 维度 | Geological object / 地质对象 | Element type / 单元类型 | Yangyi MaterialID / 阳易模型材料编号 |
|---:|---|---|---:|
| 3D | Rock matrix / 岩石基质 | Tetrahedron / 四面体 | 0 |
| 2D | Fault or fracture / 断层或裂隙 | Triangle / 三角形 | 1–10 |
| 1D | Fracture intersection / 裂隙交线 | Line / 线 | 100 |

# 2. Software installation / 软件安装

## 2.1 Recommended Linux installation / 推荐的 Linux 安装方式

Create an isolated Python environment / 创建独立的 Python 环境：

```bash
python3 -m venv porepy-env
source porepy-env/bin/activate
python -m pip install --upgrade pip
```

Download and install the stable PorePy source / 下载并安装 PorePy 稳定版本：

```bash
git clone https://github.com/pmgbergen/porepy.git
cd porepy
git checkout main
python -m pip install .
```

Install packages used for conversion and visualization / 安装网格转换与可视化所需的软件包：

```bash
python -m pip install numpy pandas meshio pyvista
```

Verify the installation / 检查安装：

```bash
python -c "import porepy, meshio, pyvista, pandas; print('Installation successful')"
gmsh --version
```

If Gmsh is not available / 如果系统中没有 Gmsh：

```bash
sudo apt update
sudo apt install gmsh
```

**English.** PorePy officially recommends Docker or a Linux source installation. Native Windows operation is not officially supported; Windows users should preferably use WSL2 or Docker.

**中文。** PorePy 官方推荐使用 Docker 或在 Linux 下从源代码安装。原生 Windows 环境并未得到官方完整支持，因此 Windows 用户最好使用 WSL2 或 Docker。

Official resources / 官方资源：

- PorePy repository / PorePy代码库: <https://github.com/pmgbergen/porepy>
- Installation instructions / 安装说明: <https://github.com/pmgbergen/porepy/blob/develop/Install.md>
- Mixed-dimensional meshing tutorial / 混合维度网格教程: <https://github.com/pmgbergen/porepy/blob/develop/tutorials/mixed_dimensional_meshing.ipynb>

# 3. Working-folder structure / 工作目录结构

The minimum project structure is / 最基本的项目目录结构为：

```text
DFNM-Mesh/
├── generate_yangyi_dfn_split_outputs.py
├── faults_all.csv
└── faults/
    ├── F3_clipped.vtp
    ├── F5_clipped.vtp
    └── ...
```

**English.** The script uses relative paths. It must be run from the folder containing both the Python script and `faults_all.csv`.

**中文。** 脚本使用相对路径，因此必须在同时包含 Python 脚本和 `faults_all.csv` 的目录中运行。

The principal Yangyi files are / 阳易模型的主要文件为：

```text
/home/zhai/Yangyi_OGS/Mesh/generate_yangyi_dfn_split_outputs.py
/home/zhai/Yangyi_OGS/Mesh/faults_all.csv
/home/zhai/Yangyi_OGS/Mesh/faults/
```

# 4. Fault-input table / 断层输入参数表

The input CSV must contain the following columns / 输入 CSV 必须包含以下字段：

```csv
fault_id,center_x,center_y,center_z,major_axis,minor_axis,major_axis_angle,strike,dip,material_id
F3,245619.25,3291286.12,4206.40,934.79,655.51,4.11,352.44,82.92,1
F5,245983.03,3292023.27,3798.43,1428.38,1137.18,7.26,348.52,75.34,3
```

| CSV field / 字段 | Explanation / 说明 |
|---|---|
| `fault_id` | Unique fault name / 唯一的断层名称 |
| `center_x`, `center_y`, `center_z` | Center of the elliptical fault / 椭圆断层面的中心坐标 |
| `major_axis` | Major semi-axis used by the script / 脚本使用的长半轴 |
| `minor_axis` | Minor semi-axis used by the script / 脚本使用的短半轴 |
| `major_axis_angle` | Rotation of the major axis within the fault plane / 长轴在断层面内的旋转角度 |
| `strike` | Strike measured clockwise from north / 从正北方向顺时针计算的走向 |
| `dip` | Downward angle measured from horizontal / 从水平面向下计算的倾角 |
| `material_id` | Final fracture MaterialID / 最终裂隙材料编号 |

Important conventions / 重要约定：

- Coordinates and axis lengths use the same unit, normally metres. / 坐标和轴长必须使用相同单位，通常为米。
- The coordinates are $x=$ east, $y=$ north and $z=$ elevation, positive upward. / 坐标约定为 $x=$ 东向、$y=$ 北向、$z=$ 高程，并且向上为正。
- Angles in the CSV are in degrees. / CSV 中的角度单位均为度。
- `major_axis` and `minor_axis` are used as semi-axes in the current script. If the source data give full lengths, divide them by two. / 当前脚本将 `major_axis` 和 `minor_axis` 作为半轴使用；如果原始数据为完整轴长，必须先除以 2。
- Confirm the geological strike convention and any required 180° correction. / 必须确认地质走向的定义，并检查是否需要进行 180° 修正。

# 5. Constructing a 3D elliptical fracture / 构建三维椭圆裂隙

**English.** For strike $\alpha$ and dip $\delta$, the script defines the horizontal strike vector and the downward dip vector as

**中文。** 对于走向 $\alpha$ 和倾角 $\delta$，脚本分别定义水平走向向量和向下倾向向量：

$$
\mathbf{s}=
\begin{bmatrix}
\sin\alpha\\
\cos\alpha\\
0
\end{bmatrix},
\qquad
\mathbf{d}=
\begin{bmatrix}
\cos\alpha\cos\delta\\
-\sin\alpha\cos\delta\\
-\sin\delta
\end{bmatrix}.
$$

For the in-plane rotation $\beta$ / 对于断层面内旋转角 $\beta$：

$$
\mathbf{u}=\cos\beta\,\mathbf{s}+\sin\beta\,\mathbf{d},
$$

$$
\mathbf{v}=-\sin\beta\,\mathbf{s}+\cos\beta\,\mathbf{d}.
$$

The ellipse is sampled using / 椭圆边界点由下式生成：

$$
\mathbf{x}(\theta)=\mathbf{c}
+a\cos\theta\,\mathbf{u}
+b\sin\theta\,\mathbf{v},
$$

where $\mathbf{c}$ is the fault center and $a$ and $b$ are the major and minor semi-axes. / 其中，$\mathbf{c}$ 为断层中心，$a$ 和 $b$ 分别为长半轴和短半轴。

The polygon is passed to PorePy using / 利用以下命令将多边形传递给 PorePy：

```python
fracture = pp.PlaneFracture(points)
```

**English.** The Yangyi script uses 64 boundary points for every ellipse. Increasing this number creates a smoother outline but may increase the cost and complexity of intersection processing.

**中文。** 阳易脚本使用 64 个边界点近似每个椭圆。增加边界点数量可使椭圆边界更加光滑，但也可能增加交线计算和网格生成的复杂度。

# 6. Model domain / 模型范围

The Yangyi bounding box is / 阳易模型的边界范围为：

```python
xmin, xmax = 243300.0, 247300.0
ymin, ymax = 3289500.0, 3293500.0
zmin, zmax = 1084.0, 5084.0
```

Create the PorePy domain / 创建 PorePy 模型域：

```python
bounding_box = {
    "xmin": xmin,
    "xmax": xmax,
    "ymin": ymin,
    "ymax": ymax,
    "zmin": zmin,
    "zmax": zmax,
}

domain = pp.Domain(bounding_box=bounding_box)
```

**English.** The domain should contain all relevant fractures. Portions outside the bounding box are clipped from the reservoir mesh. Checking only the fracture centers is insufficient because a large ellipse may extend beyond the box.

**中文。** 模型域应包含所有需要保留的裂隙。超出边界框的断层部分会被裁剪。仅检查断层中心是否位于模型域内是不够的，因为尺寸较大的椭圆仍可能超出边界框。

# 7. Create the fracture network / 创建裂隙网络

Read the table and construct the fracture list / 读取参数表并构建裂隙列表：

```python
faults = pd.read_csv("faults_all.csv")
fractures = build_fractures_from_table(faults)

network = pp.create_fracture_network(
    fractures=fractures,
    domain=domain,
)
```

At this stage PorePy / 在此阶段，PorePy 将：

- store the 3D fracture polygons; / 保存三维裂隙多边形；
- calculate fracture–fracture intersections; / 计算裂隙之间的交线；
- clip fractures against the model domain; / 根据模型域裁剪裂隙；
- prepare the geometry for conforming meshing. / 为相容网格生成准备几何模型。

# 8. Mesh-size parameters / 网格尺寸参数

The Yangyi model uses / 阳易模型采用：

```python
mesh_args = {
    "cell_size_boundary": 200.0,
    "cell_size_fracture": 80.0,
    "cell_size_min": 20.0,
    "export": True,
    "filename": "yangyi_mdg",
}
```

| Parameter / 参数 | Function / 作用 |
|---|---|
| `cell_size_boundary` | Approximate element size near the external boundary / 外部边界附近的近似单元尺寸 |
| `cell_size_fracture` | Target element size on fracture surfaces / 裂隙面上的目标单元尺寸 |
| `cell_size_min` | Minimum size near intersections and small features / 裂隙交线和细小几何特征附近的最小尺寸 |
| `export` | Retain the Gmsh intermediate files / 保留 Gmsh 中间文件 |
| `filename` | Base name of generated files / 生成文件的基本名称 |

Recommended procedure / 推荐操作：

1. Begin with a coarse mesh to identify geometry problems. / 首先使用较粗网格排查几何问题。
2. Refine the fracture surfaces gradually. / 逐步细化裂隙面。
3. Avoid an unnecessarily small `cell_size_min`. / 避免将最小尺寸设置得过小。
4. Conduct a mesh-convergence study for the final simulation. / 对最终模拟进行网格收敛性分析。

A coarse debugging configuration is / 用于调试的粗网格可设置为：

```python
mesh_args = {
    "cell_size_boundary": 400.0,
    "cell_size_fracture": 160.0,
    "cell_size_min": 40.0,
    "export": True,
    "filename": "test_mesh",
}
```

# 9. Generate the mixed-dimensional mesh / 生成混合维度网格

```python
mdg = pp.create_mdg(
    "simplex",
    mesh_args,
    network,
)
```

The `simplex` option generates / `simplex` 选项将生成：

- tetrahedra in the 3D rock matrix; / 三维岩石基质中的四面体；
- triangles on the 2D fractures; / 二维裂隙面上的三角形；
- lines along the 1D fracture intersections. / 一维裂隙交线上的线单元。

Run the complete Yangyi script with / 运行完整阳易脚本：

```bash
cd /home/zhai/Yangyi_OGS/Mesh
source /path/to/porepy-env/bin/activate
python generate_yangyi_dfn_split_outputs.py
```

**English.** Three-dimensional conforming meshing may require several minutes and substantial memory. Runtime depends strongly on the number of fractures, intersection complexity, and minimum element size.

**中文。** 三维相容网格生成可能需要数分钟并占用较多内存。计算时间主要取决于裂隙数量、交线复杂程度以及最小单元尺寸。

# 10. Convert the Gmsh mesh to VTU / 将 Gmsh 网格转换为 VTU

**English.** PorePy exports a Gmsh `.msh` file. The Yangyi script reads this file with MeshIO and creates a unified PyVista/VTU mesh. It assigns the following `grid_dim` and `MaterialIDs` values.

**中文。** PorePy 首先输出 Gmsh `.msh` 文件。阳易脚本随后利用 MeshIO 读取该文件，并创建统一的 PyVista/VTU 网格，同时分配以下 `grid_dim` 和 `MaterialIDs`。

```text
Tetrahedra / 四面体  → grid_dim = 3, MaterialID = 0
Triangles / 三角形   → grid_dim = 2, MaterialID = fault-specific value / 对应断层编号
Lines / 线单元       → grid_dim = 1, MaterialID = 100
```

**English.** In the current Yangyi script, each fracture triangle is assigned to the most likely input fault using the distance to the fault plane, its position relative to the ellipse, and agreement between the triangle normal and fault normal. Therefore, the final triangle MaterialID comes from the `material_id` field in `faults_all.csv`, rather than only from the numerical order of Gmsh physical tags.

**中文。** 在当前阳易脚本中，每个裂隙三角形根据其到断层面的距离、相对于椭圆的位置以及三角形法向与断层法向的一致性，被分配给最可能对应的输入断层。因此，最终三角形的 MaterialID 来源于 `faults_all.csv` 中的 `material_id` 字段，而不是仅由 Gmsh 物理标签的数值顺序决定。

# 11. Output files / 输出文件

```text
_out_yangyi_dfn_split/
├── gmsh_frac_file.msh
├── gmsh_frac_file.geo_unrolled
├── yangyi_unified_ogs.vtu
├── yangyi_1d_intersections.vtu
├── yangyi_2d_fractures.vtu
├── yangyi_3d_matrix.vtu
└── yangyi_2d3d_fractures_matrix.vtu
```

| File / 文件 | Purpose / 用途 |
|---|---|
| `yangyi_unified_ogs.vtu` | Unified 3D matrix, 2D fractures and 1D intersections / 统一的三维基质、二维裂隙和一维交线网格 |
| `yangyi_1d_intersections.vtu` | Fracture-intersection lines only / 仅包含裂隙交线 |
| `yangyi_2d_fractures.vtu` | Fracture triangles only / 仅包含裂隙三角形 |
| `yangyi_3d_matrix.vtu` | Matrix tetrahedra only / 仅包含基质四面体 |
| `yangyi_2d3d_fractures_matrix.vtu` | Matrix and fractures without intersection lines / 包含基质和裂隙，但不含交线 |
| `gmsh_frac_file.msh` | Original Gmsh mesh for debugging and conversion / 用于检查和转换的原始 Gmsh 网格 |

# 12. Inspect the mesh in ParaView / 在 ParaView 中检查网格

Open the unified mesh / 打开统一网格：

```bash
paraview _out_yangyi_dfn_split/yangyi_unified_ogs.vtu
```

Recommended checks / 推荐检查内容：

1. Color the mesh by `MaterialIDs`. / 使用 `MaterialIDs` 对网格着色。
2. Confirm that the matrix has MaterialID 0. / 确认基质的 MaterialID 为 0。
3. Confirm that every fracture has the intended MaterialID. / 确认每条裂隙具有正确的 MaterialID。
4. Color by `grid_dim` to distinguish 1D, 2D and 3D cells. / 使用 `grid_dim` 区分一维、二维和三维单元。
5. Apply **Threshold** to inspect one fracture at a time. / 使用 **Threshold** 逐条检查裂隙。
6. Inspect small intersection regions for distorted tetrahedra. / 检查裂隙交汇区域是否存在畸变四面体。
7. Confirm that no unexpected fragments occur outside the reservoir. / 确认储层外部没有异常的断层碎片。

To show only fracture cells / 仅显示裂隙单元：

```text
Filters → Threshold
Scalars: grid_dim
Minimum: 2
Maximum: 2
```

# 13. Topological quality checks / 拓扑质量检查

## 13.1 Fracture–matrix conformity / 裂隙与基质相容性

Every fracture triangle should coincide with a tetrahedron face / 每个裂隙三角形都应与四面体面重合：

```text
matching tetra faces = total fracture triangles
missing = 0
```

An internal fracture triangle is normally shared by two tetrahedra. / 内部裂隙三角形通常由两个四面体共享。

## 13.2 Intersection conformity / 裂隙交线相容性

Every line element should coincide with an edge of a fracture triangle / 每个交线单元都应与裂隙三角形的一条边重合：

```text
matching triangle edges = total intersection lines
missing = 0
```

**English.** A nonzero missing count indicates a topological inconsistency and the mesh should not be used directly in a coupled DFNM simulation. These tests do not assess element shape, so tetrahedral quality should also be checked in Gmsh or ParaView.

**中文。** 如果 `missing` 不为零，说明网格存在拓扑不相容问题，不应直接用于耦合 DFNM 模拟。这些检查不能评价单元形状，因此还应利用 Gmsh 或 ParaView 检查四面体的形状质量。

# 14. Common problems / 常见问题

## Gmsh fails during meshing / Gmsh 网格生成失败

Possible causes / 可能原因：

- nearly coincident fractures; / 裂隙几乎重合；
- extremely short intersection segments; / 交线过短；
- fractures touching only at a corner; / 裂隙仅在角点接触；
- an excessively small `cell_size_min`; / `cell_size_min` 过小；
- duplicate polygon points; / 多边形中存在重复点；
- the Gmsh executable is unavailable. / 系统无法调用 Gmsh。

First try a coarser mesh. If it succeeds, reduce the element sizes gradually. / 建议首先尝试较粗网格；如果成功，再逐步减小单元尺寸。

## Fault orientation is incorrect / 断层方向错误

Check whether / 检查以下内容：

- strike is measured clockwise from north; / 走向是否从正北方向顺时针计算；
- dip is measured downward from horizontal; / 倾角是否从水平面向下计算；
- `z` represents elevation rather than positive-downward depth; / `z` 是否代表高程，而不是向下为正的深度；
- a 180° strike correction is required. / 是否需要进行 180° 走向修正。

## MaterialIDs are incorrect / MaterialID 分配错误

Ensure that / 应确保：

- each fault has its intended unique `material_id`; / 每条断层具有预期的唯一 `material_id`；
- centers, strike, dip and ellipse axes are correct; / 中心、走向、倾角和椭圆轴参数正确；
- nearly coplanar or overlapping faults are checked carefully. / 对近似共面或相互重叠的断层进行重点检查。

## Mesh is too large / 网格规模过大

Increase the mesh sizes, for example / 增大网格尺寸，例如：

```python
mesh_size_boundary = 300.0
mesh_size_fracture = 120.0
mesh_size_min = 30.0
```

**English.** Halving a target size in a 3D tetrahedral mesh can increase the cell count and memory requirement dramatically.

**中文。** 对三维四面体网格而言，将目标单元尺寸减半可能显著增加单元数量和内存需求。

## The script cannot find the CSV file / 脚本找不到 CSV 文件

Run the script from its project folder / 从项目目录运行脚本：

```bash
cd /path/to/DFNM-Mesh/Mesh
python generate_yangyi_dfn_split_outputs.py
```

# 15. Generating the DFNM variants / 生成不同的 DFNM 模型

**English.** PorePy produces the base mixed-dimensional fracture–matrix mesh. The model variants used in this study are subsequently generated through material-based cell selection.

**中文。** PorePy 首先生成基础的混合维度裂隙–基质网格。本研究中的不同模型随后通过基于材料编号的单元筛选获得。

| Model / 模型 | Retained cells / 保留的单元 |
|---|---|
| Full DFNM / 完整DFNM | Matrix MaterialID 0, fractures 1–10, and LRZ 12 / 基质0、裂隙1–10和低阻区12 |
| DFNM-LRZ | Remove MaterialID 0; retain fractures and LRZ / 删除材料0，保留裂隙和低阻区 |
| DFN | Remove MaterialIDs 0 and 12; retain fracture cells / 删除材料0和12，仅保留裂隙单元 |

**English.** The low-resistivity zone represented by MaterialID 12 is introduced during a later post-processing step; it is not created by the basic PorePy fracture-network command. For final reproducibility, scripted cell extraction is preferable to manual deletion in ParaView.

**中文。** MaterialID 12 所表示的低电阻率区是在后续处理阶段加入的，并非由基础 PorePy 裂隙网络命令直接生成。为了保证最终流程可重复，建议使用脚本筛选单元，而不是仅在 ParaView 中手动删除。

# 16. Reproducibility checklist / 可重复性检查清单

Before sharing or publishing the mesh, retain / 在共享或发表网格之前，应保存：

- the generation script; / 网格生成脚本；
- the original `faults_all.csv`; / 原始 `faults_all.csv`；
- domain bounds and mesh-size parameters; / 模型范围和网格尺寸参数；
- one representative output mesh; / 一个具有代表性的输出网格；
- topology and mesh-quality reports; / 拓扑和网格质量检查结果；
- exact software versions. / 确切的软件版本。

Record the versions with / 使用以下命令记录软件版本：

```bash
python --version
gmsh --version
python -c "import porepy; print(porepy.__version__)"
python -c "import meshio; print(meshio.__version__)"
python -c "import pyvista; print(pyvista.__version__)"
```

**English.** PorePy is actively developed and its interface may change between versions. Recording the environment is therefore essential for reproducing the mesh.

**中文。** PorePy 仍在持续开发，不同版本之间的接口可能发生变化。因此，准确记录软件环境对于网格结果的重现至关重要。

# Citation / 引用

If PorePy is used in published research, cite / 如果研究中使用了 PorePy，请引用：

Keilegavlen, E., Berge, R., Fumagalli, A., Starnoni, M., Stefansson, I., Varela, J., and Berre, I. (2021). *PorePy: an open-source software for simulation of multiphysics processes in fractured porous media*. Computational Geosciences, 25, 243–265. <https://doi.org/10.1007/s10596-020-10002-5>
