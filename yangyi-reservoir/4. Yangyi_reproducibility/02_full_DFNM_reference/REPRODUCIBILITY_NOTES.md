# Full DFNM reference

The supplied source contains exactly one project: `yangyi_actual_ops_after20190113_outlier_checked.prj`; it is therefore retained as the source-supported reference. It includes matrix Material 0, faults 1–10, and volumetric LRZ Material 12. Boundary conditions are hydrostatic Dirichlet on y-min/y-max and zero Neumann on x-min/x-max; z-face meshes are registered but not assigned a boundary condition. Source terms are time-varying nodal injection at ZK403 and production at ZK208/ZK203.

The XML specifies `t_end=17,193,600 s`, 199 repeats of a one-day step. This supersedes stale names and the 290-day statement in the older source README. Source curves are embedded; the retained CSV/workbook document their derivation. Monitoring mesh points represent SC211, ZK204, ZK206, ZK207, ZK401, ZK402, and ZK501.

Baseline permeabilities (m2) by active Material ID are: 0=`1e-16`; 1=`4e-9`; 2–4=`4e-17`; 5–8=`4e-10`; 9–10=`4e-9`; and LRZ 12=`4e-11`. The project also retains an unused Material 100 definition at `4e-10`. These are exploratory/reference values, not the later `joint_007` set.

Run from this directory with `ogs yangyi_actual_ops_after20190113_outlier_checked.prj`. The comparison script requires the newly generated PVD/VTUs.
