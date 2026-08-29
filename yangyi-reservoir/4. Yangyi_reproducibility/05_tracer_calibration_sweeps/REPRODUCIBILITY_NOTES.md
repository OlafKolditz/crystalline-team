# Four-pipe tracer joint sweep

This supplied branch contains the final 20-case joint design: 16 fixed-seed Latin-hypercube points and four physical anchors. Earlier fast/storage, dispersivity, deep-recovery, and release-time branches are dependencies/history outside the supplied directory and are not reconstructed here.

The selected best coarse source case is `runs/anchor_80C/case6_joint_anchor_80C.prj`; run its documented path-only copy `case6_joint_anchor_80C_portable.prj`: explicit pipe/storage fraction 1/3, unresolved deep fraction 2/3, inferred explicit-path temperature 80 °C, fast-path longitudinal dispersivity 5 m, transverse dispersivity 0.1 m, eventual deep recovery 70%, and characteristic release time 60 d. The simulation lasts 238 d with 60 s, 900 s, and 3,600 s timestep stages. Operational and return curves are embedded in the project. Aperture is scaled by `(1/3)/(2/7)=7/6`; effective cross-sections are carried in the input mesh.

Observed tracer is 2,6-naphthalenedisulfonic acid disodium salt. Target peaks: ZK208 5 d/518.88 ppb; ZK203 32 d/164.60 ppb. Recorded selected-case predictions were 4.990 d/464.674 ppb and 32.260 d/151.213 ppb. This is an effective reduced topology; hydraulic calibration does not determine its flow partition.

The original projects preserve absolute source mesh paths. Every original is accompanied by a `*_portable.prj` whose only changes are the top-level mesh paths to local `input_mesh/` files. Run portable copies from their run directories. Regenerating/scoring the whole design with `run_joint_sweep.py` additionally needs the external deep-feedback, flow-fraction, and release-time helper scripts listed in the dependency audit.
