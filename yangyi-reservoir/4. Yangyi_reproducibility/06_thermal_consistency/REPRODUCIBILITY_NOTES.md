# Thermal consistency refinement

Nine retained projects vary M203 deep-return temperature (190, 192, 194 °C) and thermal longitudinal dispersivity (25, 30, 35 m). The selected best tested case is `runs/M203T190C_S203A0p3x_S208A6x_alphaL35m/M203T190C_S203A0p3x_S208A6x_alphaL35m.prj`.

It uses initial effective temperature 185 °C, real embedded ZK403 injection-temperature/rate curves, M208/M203 deep-return temperatures 200/190 °C, water density 1,000 kg m-3, heat capacity 4,180 J kg-1 K-1, conductivity 0.6 W m-1 K-1, and effective solid density/conductivity 2,700 kg m-3/2.5 W m-1 K-1. Effective solid heat capacities are 4,800 J kg-1 K-1 on ZK208 paths (6x), 240 on ZK203 paths (0.3x), and 800 otherwise. Thermal dispersivities are 35 m longitudinal and 0.1 m transverse.

The simulation is 1,461 d. Evaluation windows are days 12–120, 120–240, and 240–1,461. Selected-case RMSE is 5.221 °C at ZK203 and 7.437 °C at ZK208; full mean 6.329 °C and six-stage balanced 8.378 °C.

These heat capacities are lumped/effective parameters. This is an auxiliary consistency assessment on a tracer-constrained reduced topology, not a final predictive Full-DFNM thermal model. Each project runs from its run directory. Regeneration and scoring need the external parent case-3 project, workbook, and Python dependency folder.
