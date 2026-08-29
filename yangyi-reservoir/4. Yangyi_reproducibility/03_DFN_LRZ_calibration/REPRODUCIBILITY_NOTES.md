# DFN-LRZ hydraulic calibration

Sequence: 25-case coarse pressure sweep (24 log-uniform draws plus baseline); 25-case refined pressure sweep; 38-case LHS joint pressure/head sweep; 32-case local head refinement. The package retains all input projects, fixed-seed design tables, and generators, but no completed outputs or derived rankings.

Coarse/refined pressure objective: `0.60 RMSE(ZK403) + 0.20 RMSE(ZK208) + 0.20 RMSE(ZK203)`. Joint scoring equally balances normalized completion-pressure RMSE at ZK403/ZK208/ZK203 and head RMSE at SC211/ZK207/ZK206, with a 0.5 MPa per-well pressure guardrail. Final refinement fixes Materials 1 and 5, varies 12 and 9, targets SC211/ZK207, treats ZK206 diagnostically, and enforces 0.32/0.46/0.45 MPa guardrails at ZK403/ZK208/ZK203.

Final best-tested case: `random_wellhead_calibration/joint_downhole_head_calibration/head_refinement/projects/joint_007.prj`. Permeabilities (m2): matrix definition 0=`1e-16` (no matrix cells); F3/1=`9.554224501857e-9`; F5 North/2=`4e-17`; F5/3=`4e-17`; F5 South/4=`4e-17`; F8/5=`1.013370361393e-10`; F9a/6=`4e-10`; F9b/7=`4e-9`; F10/8=`4e-10`; F11a/9=`1.125163238659e-11`; F11b/10=`4e-13`; LRZ/12=`2.499956611737e-12`.

Final RMSE: pressure 0.289/0.428/0.427 MPa at ZK403/ZK208/ZK203; head 6.798 m at SC211 and 3.609 m at ZK207; diagnostic ZK206 7.272 m. This is the best tested pressure-feasible case, not proof of a unique/global optimum.

All projects use 200 daily steps. Launch nested projects from this category root so `input_mesh/...` and `source_terms/...` resolve. Sweep generators additionally reference helper scripts in a sibling base case outside the supplied source; see the dependency audit.
