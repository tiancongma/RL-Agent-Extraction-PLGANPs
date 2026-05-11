# WFDTQ4VX Dual Lineage Artifact Map

## Scope

This file records the exact governed artifact paths used for the
paper-specific dual-lineage contrast audit for `WFDTQ4VX`.

The lineages are resolved from explicit run evidence already established in the
prior WFDTQ4VX audit:

- Lineage A:
  - older frozen raw-response replay lineage used by the `2026-04-15`
    operational no-LLM replay
- Lineage B:
  - newer live/current Stage2 recovery lineage that produced `33` final rows
    for `WFDTQ4VX`

## Lineage A

### Lineage identity

- operational replay bucket:
  - `data/results/20260415_4f1c2ab/`
- Stage2 raw-response authority:
  - `data/results/20260402_5c1e7a4/01_dev15_raw_llm_freeze/frozen_raw_responses/WFDTQ4VX__stage2_v2_raw_response.json`

### Surface map

1. raw response
   - `data/results/20260402_5c1e7a4/01_dev15_raw_llm_freeze/frozen_raw_responses/WFDTQ4VX__stage2_v2_raw_response.json`
2. semantic objects
   - `data/results/20260415_4f1c2ab/01_s2_5_semantic_parsing_replay/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
3. semantic summary
   - `data/results/20260415_4f1c2ab/01_s2_5_semantic_parsing_replay/semantic_stage2_objects/semantic_stage2_v2_summary.tsv`
4. contract validation report
   - `data/results/20260415_4f1c2ab/02_s2_6_contract_validation/analysis/stage2_semantic_authority_contract_report_v1.json`
5. compatibility projection summary
   - `data/results/20260415_4f1c2ab/03_s2_7_completed_stage2/semantic_to_widerow_adapter/compatibility_projection_summary_v1.json`
6. compatibility projection trace
   - `data/results/20260415_4f1c2ab/03_s2_7_completed_stage2/semantic_to_widerow_adapter/compatibility_projection_trace_v1.tsv`
7. weak labels
   - `data/results/20260415_4f1c2ab/03_s2_7_completed_stage2/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
8. Stage3 relation records
   - `data/results/20260415_4f1c2ab/04_operational_baseline_final/formulation_relation_v1/formulation_relation_records_v1.tsv`
9. Stage3 resolved relation fields
   - `data/results/20260415_4f1c2ab/04_operational_baseline_final/formulation_relation_v1/resolved_relation_fields_v1.tsv`
10. Stage5 final table
    - `data/results/20260415_4f1c2ab/04_operational_baseline_final/final_formulation_table_v1.tsv`
11. Stage5 decision trace
    - `data/results/20260415_4f1c2ab/04_operational_baseline_final/final_output_decision_trace_v1.tsv`
12. Stage5 summary
    - `data/results/20260415_4f1c2ab/04_operational_baseline_final/final_output_summary_v1.md`
13. paper-level comparison table
    - `data/results/20260415_4f1c2ab/04_operational_baseline_final/paper_level_formulation_count_comparison.tsv`
14. identity-freeze summary
    - `data/results/20260415_4f1c2ab/04_operational_baseline_final/identity_freeze_summary_v1.tsv`

### Direct WFDTQ4VX-specific report surfaces connected to Lineage A

- replay failure summary:
  - `docs/audits/wfdtq4vx_repeat_failure_analysis.md`

## Lineage B

### Lineage identity

- governed recovery run:
  - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1`
- this run is explicitly recorded in its run context as:
  - `source_mode = live_llm`
- compare surface showing WFDTQ4VX count `33`:
  - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/analysis/final_table_vs_gt_counts.tsv`

### Surface map

1. raw response
   - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/semantic_stage2_objects/raw_responses/WFDTQ4VX__stage2_v2_raw_response.json`
2. semantic objects
   - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
3. semantic summary
   - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/semantic_stage2_objects/semantic_stage2_v2_summary.tsv`
4. contract validation report
   - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/analysis/stage2_semantic_authority_contract_report_v1.json`
5. compatibility projection summary
   - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/semantic_to_widerow_adapter/compatibility_projection_summary_v1.json`
6. compatibility projection trace
   - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/semantic_to_widerow_adapter/compatibility_projection_trace_v1.tsv`
7. weak labels
   - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
8. Stage3 relation records
   - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/formulation_relation_v1/formulation_relation_records_v1.tsv`
9. Stage3 resolved relation fields
   - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/formulation_relation_v1/resolved_relation_fields_v1.tsv`
10. Stage5 final table
    - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/final_formulation_table_v1.tsv`
11. Stage5 decision trace
    - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/final_output_decision_trace_v1.tsv`
12. Stage5 summary
    - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/final_output_summary_v1.md`
13. compare output
    - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/analysis/final_table_vs_gt_counts.tsv`
14. compare delta table
    - `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/analysis/dev15_rebuilt_stage2_compare_table_v1.tsv`

### Direct WFDTQ4VX-specific report surfaces connected to Lineage B

- persistence validation at the live-call boundary:
  - `data/results/20260414_0011ee7/11_s2_4b_wfdtq4vx_persistence_validation/analysis/s2_4b_request_summary_v1.tsv`
- paper-local post-selector S2-5 replay check:
  - `data/results/20260414_0011ee7/23_wfdtq4vx_post_selector_marker_check_s2_5_v1/semantic_stage2_objects/semantic_stage2_v2_summary.tsv`
- paper-local S2-7 downstream continuation:
  - `data/results/20260414_0011ee7/29_wfdtq4vx_post_selector_downstream_s2_7_v1/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- paper-local Stage2 recovery audit:
  - `data/results/20260410_a165cd1/11_s2_dev15_full_freeze_v1/analysis/wfdtq4vx_doe_recovery_final_report.md`
- prior repo audit:
  - `docs/audits/wfdtq4vx_fix_audit.md`

## Contrast note

The most important path contrast is:

- Lineage A raw-response authority comes from:
  - `2026-04-02` frozen replay surface
- Lineage B recovery comes from:
  - `2026-04-14` live/current Stage2 full rebuild surface

That is an explicit lineage split, not a recency guess.
