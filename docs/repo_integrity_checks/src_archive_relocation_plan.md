# 1. Planning scope

This is a dry-run family-level relocation plan for `src/` only.
No files were moved, deleted, renamed, or behavior-modified.
The goal is to identify historical script families that should later be relocated out of the current main stage directories so the active/stable layer is easier to read.

# 2. Families that should remain in main stage directories

Only families that still clearly belong to the current active or stable engineering layer should remain visually mixed with the main stage directories.

## 1. Stage0 relevance acquisition and Zotero sync

Representative scripts:

- `src/stage0_relevance/auto_tag_plga_gemini.py`
- `src/stage0_relevance/auto_tag_plga_openai.py`
- `src/stage0_relevance/classify_gemini_grouped.py`
- `src/stage0_relevance/fill_missing_snapshots.py`
- `src/stage0_relevance/prefilter_regex.py`
- `src/stage0_relevance/zotero_api_sync_selected.py`
- `src/stage0_relevance/zotero_fetch_llm_relevant_pdfs.py`
- `src/stage0_relevance/zotero_llm_relevant_interactive.py`
- `src/stage0_relevance/zotero_tag_sync.py`

Reason:
This is still the maintained raw-intake/relevance layer named in the script map.

## 2. Stage1 cleaning and table extraction

Representative scripts:

- `src/stage1_cleaning/clean_manifest_to_text.py`
- `src/stage1_cleaning/extract_tables_for_keys_v1.py`
- `src/stage1_cleaning/pdf2clean.py`
- `src/stage1_cleaning/run_tables_extraction_for_dataset_v1.py`
- `src/stage1_cleaning/zotero_raw_to_manifest.py`

Reason:
These scripts still support the maintained cleaned-text/table asset path used by the active extractor.

## 3. Active Stage2 extraction and adjacent evidence support

Representative scripts:

- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
- `src/stage2_sampling_labels/build_evidence_bundle_for_keys_v1.py`
- `src/stage2_sampling_labels/build_key2txt_from_sample_manifest.py`
- `src/stage2_sampling_labels/export_blockpack_audit_v7pilot_r3_fixparse.py`
- `src/stage2_sampling_labels/export_evidence_bundle_audit_xlsx_v1.py`
- `src/stage2_sampling_labels/sample_from_manifest_html_first.py`

Reason:
These remain adjacent to the active extractor and support the current LLM-first extraction path.

## 4. Current DEV15 evaluation and reviewer-workbook family

Representative scripts:

- `src/stage4_eval/eval_weak_labels_v7pilot3.py`
- `src/stage4_eval/build_dev15_review_workbook_v1.py`

Reason:
These are the only explicitly active Stage4 entrypoints in the runbook/flow.

## 5. Current downstream validation and evidence-QC family

Representative scripts:

- `src/stage4_eval/compute_formulation_alignment_v1.py`
- `src/stage4_eval/compute_set_level_ee_match_v1.py`
- `src/stage5_benchmark/run_evidence_realign_v1.py`
- `src/stage5_benchmark/run_evidence_token_qc_v1.py`

Reason:
These fit the current architecture: semantic formulation identification upstream, deterministic validation/audit downstream.

## 6. Stage5 schema / projection / export pipeline family

Representative scripts:

- `src/stage5_benchmark/build_two_table_schema_v2.py`
- `src/stage5_benchmark/build_two_table_schema_v3.py`
- `src/stage5_benchmark/derive_doe_coded_factors_v1.py`
- `src/stage5_benchmark/export_full_database_v1.py`
- `src/stage5_benchmark/formulation_core_signature_v1.py`
- `src/stage5_benchmark/run_alignment_eval_core_v1.py`
- `src/stage5_benchmark/run_alignment_eval_schema_v3_v1.py`
- `src/stage5_benchmark/run_alignment_eval_v1.py`
- `src/stage5_benchmark/run_core_eval_pipeline_v1.py`
- `src/stage5_benchmark/run_derivation_v1.py`
- `src/stage5_benchmark/run_formulation_core_signature_v1.py`
- `src/stage5_benchmark/run_projection_core_to_curated_v1.py`
- `src/stage5_benchmark/run_projection_to_curated_v1.py`
- `src/stage5_merge_publish/merge_results.py`

Reason:
This is the clearest ongoing downstream schema/evaluation/export layer.

## 7. Shared runtime infrastructure utilities

Representative scripts:

- `src/utils/build_dataset_split_dev_v1.py`
- `src/utils/build_dataset_split_test_v1.py`
- `src/utils/build_global_zotero_index_v1.py`
- `src/utils/gemini_models.py`
- `src/utils/html_parser.py`
- `src/utils/model_policy.py`
- `src/utils/paths.py`
- `src/utils/run_id.py`
- `src/utils/run_latest.py`
- `src/utils/run_preflight.py`
- `src/utils/split_registry_v1.py`
- `src/utils/validate_dataset_layout_v1.py`

Reason:
These are infrastructure modules/utilities rather than branch-specific historical methods.

# 3. Families recommended for archive relocation

## Family: old_gt_arbitration

Member scripts:

- `src/stage3_gt/build_gt_template_from_conflict_queue.py`
- `src/stage3_gt/export_gt_annotation_view.py`
- `src/stage3_gt/merge_gt_from_annotation_view.py`
- `src/stage3_gt/gt_summary_report.py`
- `src/legacy/20260202/gt_tool.py`
- `src/legacy/20260202/gt_tool_v3.py`

Why the family is historical:
The confirmed project direction says the old dual-model conflict-comparison to human-GT arbitration workflow is not the default GT path anymore. These scripts came from that older arbitration-centered workflow.

Why it should not remain visually mixed with current mainline scripts:
Keeping them in `src/stage3_gt/` makes the repository imply that conflict-queue arbitration remains a maintained default GT layer. Current governance no longer says that.

Recommended destination:
- `src/archive_methods/old_gt_arbitration/`

Relocation priority:
- high

## Family: dev15_skeleton_bootstrap

Member scripts:

- `src/stage3_gt/build_dev15_formulation_skeleton_review_v1.py`
- `src/stage3_gt/export_dev15_formulation_skeleton_gt_v1.py`
- `src/stage3_gt/formulation_skeleton_common.py`
- `src/stage3_gt/validate_dev15_formulation_skeleton_review_v1.py`

Why the family is historical:
These scripts are tied to the bounded DEV15 skeleton-bootstrap workflow, not to the active extraction/evaluation mainline itself.

Why it should not remain visually mixed with current mainline scripts:
The current mainline keeps only a limited DEV15 formulation-count GT workflow alive. The skeleton-bootstrap family is narrower and easier to mistake for current default GT machinery if left beside active scripts.

Recommended destination:
- `src/archive_methods/dev15_skeleton_bootstrap/`

Relocation priority:
- medium

## Family: dual_model_extraction_comparison

Member scripts:

- `src/stage4_eval/auto_extract_multimodel.py`
- `src/stage4_eval/multi_model_extract_tier1.py`
- `src/stage4_eval/multi_model_extract_tier2.py`
- `src/stage4_eval/multi_model_merge_qc.py`
- `src/stage4_eval/multi_model_consensus_vote.py`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v5_G3.py`

Why the family is historical:
Confirmed project decisions state that the current mainline does not use the old dual-model conflict-comparison workflow as a default path.

Why it should not remain visually mixed with current mainline scripts:
Leaving these files next to active extractors/evaluators makes the repository look like multimodel consensus is still the architectural center. It is not.

Recommended destination:
- `src/archive_methods/dual_model_extraction_comparison/`

Relocation priority:
- high

## Family: stage4_rule_heavy_formulation_reconstruction

Member scripts:

- `src/stage4_eval/apply_formulation_grouping_v1.py`
- `src/stage4_eval/apply_global_baseline_inheritance_and_rerun_alignment_v1.py`
- `src/stage4_eval/build_boundary_alignment_diagnostics_pack_v1.py`
- `src/stage4_eval/build_failure_profile_v1.py`
- `src/stage4_eval/build_per_doi_diagnostics_v1.py`
- `src/stage4_eval/compare_drugname_sets_v1.py`
- `src/stage4_eval/compute_alignment_sensitivity_v1.py`
- `src/stage4_eval/compute_goren_metrics_tables_v1.py`
- `src/stage4_eval/run_alignment_v3_surfactant_drugnorm.py`

Why the family is historical:
Architecture now prefers LLM semantic formulation identification upstream. Rule-based logic is supposed to validate, audit, and evidence-check downstream results, not act as the primary grouping/reconstruction mechanism.

Why it should not remain visually mixed with current mainline scripts:
These files visually compete with the active Stage4 evaluator and make a rule-heavy reconstruction path appear current when governance says it is not.

Recommended destination:
- `src/archive_methods/stage4_rule_heavy_formulation_reconstruction/`

Relocation priority:
- high

## Family: benchmark_specific_audit_report

Member scripts:

- `src/stage4_eval/audit_doi_raw_field_mapping_v7pilot.py`
- `src/stage4_eval/build_doi_value_flow_audit_v7pilot.py`
- `src/stage4_eval/build_goren_overlap_scaffold_v1.py`
- `src/stage4_eval/build_v7pilot_field_mapping_audit.py`
- `src/stage4_eval/compare_v7pilot_scope_r1_r2.py`
- `src/stage4_eval/export_dev15_dashboard_and_audit_pack_v1.py`
- `src/stage4_eval/export_dev15_formulation_view_xlsx_v1.py`
- `src/stage4_eval/test_doe_coordinate_reconciliation_v1.py`
- `src/stage5_benchmark/analyze_row_membership_core_v1.py`
- `src/stage5_benchmark/analyze_row_membership_v1.py`
- `src/stage5_benchmark/build_goren_overlap_manifest.py`
- `src/stage5_benchmark/copy_goren_dataset.py`
- `src/stage5_benchmark/evaluate_against_goren.py`
- `src/stage5_benchmark/export_audit_to_excel_v1.py`
- `src/stage5_benchmark/make_audit_pack_v1.py`
- `src/stage5_benchmark/report_schema_v2_v3_core_diff_v1.py`

Why the family is historical:
These are benchmark-, report-, or audit-pack-specific paths tied to particular experimental checkpoints, reviewer packs, or comparison exercises.

Why it should not remain visually mixed with current mainline scripts:
They crowd the active evaluation/export surface and make narrow benchmark work look like general maintained pipeline infrastructure.

Recommended destination:
- `src/archive_methods/benchmark_specific_audit_report/`

Relocation priority:
- medium

## Family: older_weak_label_pilot_variants

Member scripts:

- `src/stage2_sampling_labels/auto_extract_weak_labels.py`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v3.py`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v4.py`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v5.py`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v6.py`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot.py`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r2.py`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3.py`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixflat.py`

Why the family is historical:
The active extractor is now `auto_extract_weak_labels_v7pilot_r3_fixparse.py`. The older versions are baseline or pilot lineage, not current entrypoints.

Why it should not remain visually mixed with current mainline scripts:
The current Stage2 directory is hard to read when every older extractor lineage sits beside the active entrypoint.

Recommended destination:
- `src/archive_methods/older_weak_label_pilot_variants/`

Relocation priority:
- high

## Family: legacy_pre_mainline_snapshots

Member scripts:

- `src/legacy/20260130/csv2clean_manifest.py`
- `src/legacy/20260130/pdf2clean.py`
- `src/legacy/20260130/zotero_csv_to_manifest_tsv.py`
- `src/legacy/20260131/clean_manifest_to_text.py`
- `src/legacy/20260131/sample_from_manifest_html_first.py`

Why the family is historical:
These are already legacy snapshots of preprocessing/sampling paths superseded by maintained stage scripts.

Why it should not remain visually mixed with current mainline scripts:
If the repository keeps using `src/legacy/`, it should organize it by historical family instead of by date snapshots alone.

Recommended destination:
- `src/legacy/pre_mainline_snapshots/`

Relocation priority:
- low

# 4. Explicit ruling on old GT arbitration path

The following scripts should be relocated as archived methods, not left in the main stage directories as if they were current GT infrastructure:

- `src/stage3_gt/build_gt_template_from_conflict_queue.py`
- `src/stage3_gt/export_gt_annotation_view.py`
- `src/stage3_gt/merge_gt_from_annotation_view.py`
- `src/stage3_gt/gt_summary_report.py`

Ruling:
These are not current mainline.

Reason:
- The confirmed project decision says the old dual-model conflict-comparison to human-GT arbitration workflow is not the default GT path.
- The current meaningful GT workflow is limited and DEV15 formulation-count-oriented.
- Therefore these scripts should not remain presented as the default GT layer unless governance explicitly reactivates them.

Recommended treatment:
- relocate with the `old_gt_arbitration` family rather than keep them in `src/stage3_gt/`

# 5. Explicit ruling on old Stage4 rule-heavy formulation reconstruction path

The following scripts should be relocated as archived methods because the architecture now prefers LLM semantic formulation identification upstream and deterministic rules for downstream validation/audit only:

- `src/stage4_eval/apply_formulation_grouping_v1.py`
- `src/stage4_eval/apply_global_baseline_inheritance_and_rerun_alignment_v1.py`
- `src/stage4_eval/build_boundary_alignment_diagnostics_pack_v1.py`
- `src/stage4_eval/build_failure_profile_v1.py`
- `src/stage4_eval/build_per_doi_diagnostics_v1.py`
- `src/stage4_eval/compare_drugname_sets_v1.py`
- `src/stage4_eval/compute_alignment_sensitivity_v1.py`
- `src/stage4_eval/compute_goren_metrics_tables_v1.py`
- `src/stage4_eval/run_alignment_v3_surfactant_drugnorm.py`
- `src/stage4_eval/build_goren_overlap_scaffold_v1.py`
- `src/stage4_eval/compare_v7pilot_scope_r1_r2.py`

Reason:
These scripts belong to the older family where deterministic rules carried more of the formulation grouping/reconstruction burden. That is no longer the preferred architectural direction.

Not relocated by this ruling:

- `src/stage4_eval/eval_weak_labels_v7pilot3.py`
- `src/stage4_eval/build_dev15_review_workbook_v1.py`
- `src/stage4_eval/compute_formulation_alignment_v1.py`
- `src/stage4_eval/compute_set_level_ee_match_v1.py`

These still fit the current active or downstream validation/audit layer.

# 6. Delete-review subset

These should remain deletion-review items rather than archive-relocation items.

- `src/stage4_eval/audit_top3_doi_root_cause_v1.py`
  - why deletion review is stronger than archive relocation: this is a one-off `top3` root-cause script with frozen defaults and weak evidence of durable historical-method value.
- `src/stage4_eval/inspect_formulation_view_inventory_v1.py`
  - why deletion review is stronger than archive relocation: this looks like unregistered inspection residue rather than a coherent historical family member.
- `src/stage5_benchmark/debug_doi_overlap.py`
  - why deletion review is stronger than archive relocation: explicit debug identity, frozen overlap defaults, and no sign that it represents a maintained historical method family.
- `src/utils/test_gemini.py`
  - why deletion review is stronger than archive relocation: simple connectivity probe with little provenance value beyond ad hoc environment testing.

# 7. Proposed execution order

1. Relocate the highest-priority historical families first:
   - `old_gt_arbitration`
   - `dual_model_extraction_comparison`
   - `stage4_rule_heavy_formulation_reconstruction`
   - `older_weak_label_pilot_variants`
2. Re-scan the repository for stale references after those family moves.
3. Relocate medium-priority benchmark/report families next.
4. Normalize the remaining `src/legacy/` subtree into family-oriented archive layout.
5. Only after relocation and reference re-scan, review the low-confidence delete candidates for possible removal.

# 8. Summary counts

- families staying in main stage directories: 7
- families recommended for archive relocation: 7
- scripts in archive-relocation scope: 55
- scripts in delete-review scope: 4
