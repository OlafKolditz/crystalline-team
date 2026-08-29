# Yangyi actual injection-production simulation + monitoring comparison

This package uses the uploaded workbook `yangyi_inj_monitor_processed_updated.xlsx` to drive the OGS source/sink terms with time-dependent curves.

## Main files

- `yangyi_actual_ops_inj_pro.prj`  
  OGS project file with time-dependent source/sink curves.
- `actual_injection_production_curves_for_ogs_cleaned.csv`  
  Cleaned source/sink curve table used by the PRJ.
- `actual_injection_production_curves_raw_from_excel.csv`  
  Raw curve table copied from the Excel sheet for checking.
- `observed_monitoring_delta_head.csv`  
  Observed monitoring-well head changes from the Excel sheet.
- `compare_simulated_observed_head_change.py`  
  Post-processing script to calculate simulated head change and compare it with observations.

## Important data treatment

1. Simulation start time is 2018-10-14, i.e. `time = 0 d`.
2. Observed monitoring head change is zeroed at 2018-10-17, i.e. `time = 3 d` from injection start.
3. The comparison script therefore calculates simulated head change as:

   `Δh_sim(t) = h_sim(t) - h_sim(3 d)`

4. One obvious outlier was corrected in the OGS curve table:

   `2019-07-01 Reinj_Q_tph_avg: 7712 t/h -> 771.2 t/h`

   The three source/sink rates on that day were divided by 10 accordingly. The raw table is retained for checking.

5. Rows present in the operational table but with missing source/sink flow values were treated as zero flow.

## Source/sink points

- ZK403 injection: `source_terms/ZK403_single_source_sink_bulk_node.vtu`
- ZK208 production: `source_terms/ZK208_single_source_sink_bulk_node.vtu`
- ZK203 production: `source_terms/ZK203_single_source_sink_bulk_node.vtu`

The production split follows the workbook assumption:

`ZK203 : ZK208 = 217 : 433`

## Run OGS

```bash
ogs yangyi_actual_ops_inj_pro.prj
```

The output prefix is:

```text
Yangyi_actual_ops_inj_pro
```

## Compare simulated and observed head change

```bash
python compare_simulated_observed_head_change.py   --pvd Yangyi_actual_ops_inj_pro.pvd   --monitor_csv monitoring/recommended_monitoring_points_all7_ZK501_bedrock.csv   --observed_csv observed_monitoring_delta_head.csv   --out_dir comparison_head_change
```

Outputs:

- `comparison_head_change/simulated_head_change_all_outputs.csv`
- `comparison_head_change/simulated_vs_observed_delta_head_at_observation_times.csv`
- `comparison_head_change/comparison_error_summary_by_well.csv`
- `comparison_head_change/compare_delta_head_all_wells.png`
- one comparison figure for each monitoring well.

## Notes

- The PRJ keeps the same mesh, medium properties, source/sink node files, and 7-monitor-well files from the previous package.
- Time-dependent curves are stored directly in the PRJ under `<curves>` and are referenced by `CurveScaled` parameters.
- Time stepping is daily from 0 to 290 days.
- Fixed output times include all monitoring dates, so the comparison script can align simulated and observed curves cleanly.
