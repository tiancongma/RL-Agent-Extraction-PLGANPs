# 1. Families relocated

Relocated to `src/archive_methods/` in this pass:

- `old_gt_arbitration`
- `dual_model_extraction_comparison`
- `stage4_rule_heavy_formulation_reconstruction`
- `older_weak_label_pilot_variants`
- `benchmark_specific_audit_report`
- `dev15_skeleton_bootstrap`

Conservative overlap resolution applied:

- `src/stage4_eval/build_goren_overlap_scaffold_v1.py`
- `src/stage4_eval/compare_v7pilot_scope_r1_r2.py`

These appeared in both the rule-heavy and benchmark-specific family approvals. They were moved once, into `src/archive_methods/benchmark_specific_audit_report/`, because that destination is the narrower and more specific family fit.

# 2. Files moved

## old_gt_arbitration

- `src/stage3_gt/build_gt_template_from_conflict_queue.py` -> `src/archive_methods/old_gt_arbitration/build_gt_template_from_conflict_queue.py`
- `src/stage3_gt/export_gt_annotation_view.py` -> `src/archive_methods/old_gt_arbitration/export_gt_annotation_view.py`
- `src/stage3_gt/merge_gt_from_annotation_view.py` -> `src/archive_methods/old_gt_arbitration/merge_gt_from_annotation_view.py`
- `src/stage3_gt/gt_summary_report.py` -> `src/archive_methods/old_gt_arbitration/gt_summary_report.py`
- `src/legacy/20260202/gt_tool.py` -> `src/archive_methods/old_gt_arbitration/gt_tool.py`
- `src/legacy/20260202/gt_tool_v3.py` -> `src/archive_methods/old_gt_arbitration/gt_tool_v3.py`

## dual_model_extraction_comparison

- `src/stage4_eval/auto_extract_multimodel.py` -> `src/archive_methods/dual_model_extraction_comparison/auto_extract_multimodel.py`
- `src/stage4_eval/multi_model_extract_tier1.py` -> `src/archive_methods/dual_model_extraction_comparison/multi_model_extract_tier1.py`
- `src/stage4_eval/multi_model_extract_tier2.py` -> `src/archive_methods/dual_model_extraction_comparison/multi_model_extract_tier2.py`
- `src/stage4_eval/multi_model_merge_qc.py` -> `src/archive_methods/dual_model_extraction_comparison/multi_model_merge_qc.py`
- `src/stage4_eval/multi_model_consensus_vote.py` -> `src/archive_methods/dual_model_extraction_comparison/multi_model_consensus_vote.py`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v5_G3.py` -> `src/archive_methods/dual_model_extraction_comparison/auto_extract_weak_labels_v5_G3.py`

## stage4_rule_heavy_formulation_reconstruction

- `src/stage4_eval/apply_formulation_grouping_v1.py` -> `src/archive_methods/stage4_rule_heavy_formulation_reconstruction/apply_formulation_grouping_v1.py`
- `src/stage4_eval/apply_global_baseline_inheritance_and_rerun_alignment_v1.py` -> `src/archive_methods/stage4_rule_heavy_formulation_reconstruction/apply_global_baseline_inheritance_and_rerun_alignment_v1.py`
- `src/stage4_eval/build_boundary_alignment_diagnostics_pack_v1.py` -> `src/archive_methods/stage4_rule_heavy_formulation_reconstruction/build_boundary_alignment_diagnostics_pack_v1.py`
- `src/stage4_eval/build_failure_profile_v1.py` -> `src/archive_methods/stage4_rule_heavy_formulation_reconstruction/build_failure_profile_v1.py`
- `src/stage4_eval/build_per_doi_diagnostics_v1.py` -> `src/archive_methods/stage4_rule_heavy_formulation_reconstruction/build_per_doi_diagnostics_v1.py`
- `src/stage4_eval/compare_drugname_sets_v1.py` -> `src/archive_methods/stage4_rule_heavy_formulation_reconstruction/compare_drugname_sets_v1.py`
- `src/stage4_eval/compute_alignment_sensitivity_v1.py` -> `src/archive_methods/stage4_rule_heavy_formulation_reconstruction/compute_alignment_sensitivity_v1.py`
- `src/stage4_eval/compute_goren_metrics_tables_v1.py` -> `src/archive_methods/stage4_rule_heavy_formulation_reconstruction/compute_goren_metrics_tables_v1.py`
- `src/stage4_eval/run_alignment_v3_surfactant_drugnorm.py` -> `src/archive_methods/stage4_rule_heavy_formulation_reconstruction/run_alignment_v3_surfactant_drugnorm.py`

## older_weak_label_pilot_variants

- `src/stage2_sampling_labels/auto_extract_weak_labels.py` -> `src/archive_methods/older_weak_label_pilot_variants/auto_extract_weak_labels.py`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v3.py` -> `src/archive_methods/older_weak_label_pilot_variants/auto_extract_weak_labels_v3.py`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v4.py` -> `src/archive_methods/older_weak_label_pilot_variants/auto_extract_weak_labels_v4.py`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v5.py` -> `src/archive_methods/older_weak_label_pilot_variants/auto_extract_weak_labels_v5.py`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v6.py` -> `src/archive_methods/older_weak_label_pilot_variants/auto_extract_weak_labels_v6.py`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot.py` -> `src/archive_methods/older_weak_label_pilot_variants/auto_extract_weak_labels_v7pilot.py`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r2.py` -> `src/archive_methods/older_weak_label_pilot_variants/auto_extract_weak_labels_v7pilot_r2.py`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3.py` -> `src/archive_methods/older_weak_label_pilot_variants/auto_extract_weak_labels_v7pilot_r3.py`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixflat.py` -> `src/archive_methods/older_weak_label_pilot_variants/auto_extract_weak_labels_v7pilot_r3_fixflat.py`

## benchmark_specific_audit_report

- `src/stage4_eval/audit_doi_raw_field_mapping_v7pilot.py` -> `src/archive_methods/benchmark_specific_audit_report/audit_doi_raw_field_mapping_v7pilot.py`
- `src/stage4_eval/build_doi_value_flow_audit_v7pilot.py` -> `src/archive_methods/benchmark_specific_audit_report/build_doi_value_flow_audit_v7pilot.py`
- `src/stage4_eval/build_goren_overlap_scaffold_v1.py` -> `src/archive_methods/benchmark_specific_audit_report/build_goren_overlap_scaffold_v1.py`
- `src/stage4_eval/build_v7pilot_field_mapping_audit.py` -> `src/archive_methods/benchmark_specific_audit_report/build_v7pilot_field_mapping_audit.py`
- `src/stage4_eval/compare_v7pilot_scope_r1_r2.py` -> `src/archive_methods/benchmark_specific_audit_report/compare_v7pilot_scope_r1_r2.py`
- `src/stage4_eval/export_dev15_dashboard_and_audit_pack_v1.py` -> `src/archive_methods/benchmark_specific_audit_report/export_dev15_dashboard_and_audit_pack_v1.py`
- `src/stage4_eval/export_dev15_formulation_view_xlsx_v1.py` -> `src/archive_methods/benchmark_specific_audit_report/export_dev15_formulation_view_xlsx_v1.py`
- `src/stage4_eval/test_doe_coordinate_reconciliation_v1.py` -> `src/archive_methods/benchmark_specific_audit_report/test_doe_coordinate_reconciliation_v1.py`
- `src/stage5_benchmark/analyze_row_membership_core_v1.py` -> `src/archive_methods/benchmark_specific_audit_report/analyze_row_membership_core_v1.py`
- `src/stage5_benchmark/analyze_row_membership_v1.py` -> `src/archive_methods/benchmark_specific_audit_report/analyze_row_membership_v1.py`
- `src/stage5_benchmark/build_goren_overlap_manifest.py` -> `src/archive_methods/benchmark_specific_audit_report/build_goren_overlap_manifest.py`
- `src/stage5_benchmark/copy_goren_dataset.py` -> `src/archive_methods/benchmark_specific_audit_report/copy_goren_dataset.py`
- `src/stage5_benchmark/evaluate_against_goren.py` -> `src/archive_methods/benchmark_specific_audit_report/evaluate_against_goren.py`
- `src/stage5_benchmark/export_audit_to_excel_v1.py` -> `src/archive_methods/benchmark_specific_audit_report/export_audit_to_excel_v1.py`
- `src/stage5_benchmark/make_audit_pack_v1.py` -> `src/archive_methods/benchmark_specific_audit_report/make_audit_pack_v1.py`
- `src/stage5_benchmark/report_schema_v2_v3_core_diff_v1.py` -> `src/archive_methods/benchmark_specific_audit_report/report_schema_v2_v3_core_diff_v1.py`

## dev15_skeleton_bootstrap

- `src/stage3_gt/build_dev15_formulation_skeleton_review_v1.py` -> `src/archive_methods/dev15_skeleton_bootstrap/build_dev15_formulation_skeleton_review_v1.py`
- `src/stage3_gt/export_dev15_formulation_skeleton_gt_v1.py` -> `src/archive_methods/dev15_skeleton_bootstrap/export_dev15_formulation_skeleton_gt_v1.py`
- `src/stage3_gt/formulation_skeleton_common.py` -> `src/archive_methods/dev15_skeleton_bootstrap/formulation_skeleton_common.py`
- `src/stage3_gt/validate_dev15_formulation_skeleton_review_v1.py` -> `src/archive_methods/dev15_skeleton_bootstrap/validate_dev15_formulation_skeleton_review_v1.py`

# 3. Files deleted

Approved low-risk deletions executed:

- `src/utils/test_gemini.py`
- `src/stage5_benchmark/debug_doi_overlap.py`

# 4. Any references updated

Updated unambiguous path/import references in active files and relocated runtime files.

Files updated:

- `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `project/PIPELINE_SCRIPT_MAP.md`
- `docs/tool_index.md`
- `docs/methods/dev15_formulation_skeleton_annotation_tool.md`
- `src/stage5_benchmark/run_core_eval_pipeline_v1.py`
- `src/archive_methods/dev15_skeleton_bootstrap/build_dev15_formulation_skeleton_review_v1.py`
- `src/archive_methods/dev15_skeleton_bootstrap/export_dev15_formulation_skeleton_gt_v1.py`
- `src/archive_methods/dev15_skeleton_bootstrap/validate_dev15_formulation_skeleton_review_v1.py`
- `src/archive_methods/benchmark_specific_audit_report/audit_doi_raw_field_mapping_v7pilot.py`
- `src/archive_methods/benchmark_specific_audit_report/build_doi_value_flow_audit_v7pilot.py`

Reference update summary:

- updated active governance docs to point at `src/archive_methods/...` for relocated historical scripts
- updated `src/stage5_benchmark/run_core_eval_pipeline_v1.py` to call the relocated `analyze_row_membership_core_v1.py`
- updated relocated DEV15 skeleton scripts to import `src.archive_methods.dev15_skeleton_bootstrap.formulation_skeleton_common`
- updated relocated benchmark audit scripts to import `src.archive_methods.older_weak_label_pilot_variants.auto_extract_weak_labels_v7pilot_r3_fixflat`
- removed deleted-file inventory rows from `docs/tool_index.md`

Approximate reference edits applied:

- `130` path/import reference replacements across the updated files above

# 5. Any unresolved ambiguities

None remain.

Notes:

- The duplicate family assignment for `build_goren_overlap_scaffold_v1.py` and `compare_v7pilot_scope_r1_r2.py` was resolved conservatively by placing them in the narrower `benchmark_specific_audit_report` family.
- Historical references remain inside archived scripts and historical docs where they do not affect active behavior.

# 6. Final note on whether active stage directories are now cleaner

Yes.

The active stage directories are materially cleaner now:

- old GT arbitration tooling no longer sits beside the bounded DEV15 workflow
- multimodel and rule-heavy reconstruction scripts no longer visually compete with the active extractor/evaluator path
- benchmark-specific audit/report scripts no longer crowd the active Stage4/Stage5 directories
- the remaining visible stage scripts better match the current LLM-first extraction plus downstream validation/audit architecture
