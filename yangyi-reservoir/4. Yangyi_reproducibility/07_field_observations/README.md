# Field observations and operational forcing

| File | Quantity and role | Provenance/use |
|---|---|---|
| `hydraulic/yangyi_after_20190113_for_ogs.xlsx` | active-well operations and monitoring-well hydraulic observations | forcing and head evaluation used by Full DFNM, DFN-LRZ, and structural comparison |
| `hydraulic/clean_six_line_pressure_data.xlsx` | processed monitored wellhead pressures | transformed/evaluated against completion-node pressure at ZK403, ZK208, ZK203 |
| `hydraulic/observed_monitoring_delta_head.csv` | processed monitoring-well head change | Full-DFNM evaluation; preserve as head change, not pressure |
| `hydraulic/ogs_source_curves_after20190113_outlier_checked.csv` | processed ZK403 injection and ZK208/ZK203 production histories | operational forcing provenance; curves are embedded in the project XML |
| `tracer/observed_tracer_ZK203_ZK208.csv` | processed long-term concentrations and sample times | tracer calibration/evaluation at ZK208 and ZK203 |

The hydraulic workbook and processed CSVs are both retained because the processed representations are directly used by project/scoring workflows. Wellhead pressure, completion-depth absolute pressure, and hydraulic-head change remain explicitly distinct.

No original thermal observation workbook occurs in the supplied `case4_local_joint_refinement_4y` directory. Its scripts refer to external `case3_split_storage_heat_exchange_4y/Power output and PT.xls`. Injection-temperature and rate curves required merely to run the selected thermal project are embedded in its XML, but independent evaluation against production-temperature observations is not possible from this package alone.
