# 1. Scope

This is a dry-run source-layer reduction audit for `src/`.
No files were deleted, moved, renamed, or behavior-modified.
The audit classifies Python scripts by engineering identity using current governance evidence from:

- `project/0_PROJECT_CHARTER.md`
- `project/1_REQUIREMENTS.md`
- `project/2_ARCHITECTURE.md`
- `project/PIPELINE_SCRIPT_MAP.md`
- `project/ACTIVE_PIPELINE_FLOW.md`
- `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `project/FILE_NAMING_AND_VERSIONING.md`
- `docs/tool_index.md`

Non-script files under `src/` such as `src/src_README.md` and `src/stage2_sampling_labels/PIPELINE_SPEC_sampling_update.md` were not classified as executable scripts.

# 2. Active mainline entrypoints

| path | why it is current mainline | evidence from active flow/runbook |
|---|---|---|
| `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py` | Current Stage2 extractor for the active DEV-15 formulation-instance path. | Named in `project/ACTIVE_PIPELINE_FLOW.md` steps `DEV15_01` and `DEV15_03`; named `ACTIVE_MAINLINE` in `project/ACTIVE_PIPELINE_RUNBOOK.md`; explicitly called the Stage2 default in `project/PIPELINE_SCRIPT_MAP.md`. |
| `src/stage4_eval/eval_weak_labels_v7pilot3.py` | Current Stage4 evaluator and reconciliation seam for official DEV counting. | Named in `project/ACTIVE_PIPELINE_FLOW.md` steps `DEV15_02` and `DEV15_04`; named `ACTIVE_MAINLINE` in `project/ACTIVE_PIPELINE_RUNBOOK.md`; explicitly called the current Stage4 DEV evaluator in `project/PIPELINE_SCRIPT_MAP.md`. |
| `src/stage4_eval/build_dev15_review_workbook_v1.py` | Active reviewer-workbook builder used at the end of the documented DEV-15 flow. It is not the evaluator itself, but it is still an active flow entrypoint. | Named in `project/ACTIVE_PIPELINE_FLOW.md` step `DEV15_05`; named `ACTIVE_SUPPORTING` in `project/ACTIVE_PIPELINE_RUNBOOK.md`; explicitly identified as the current reviewer workbook builder in `project/PIPELINE_SCRIPT_MAP.md`. |

# 3. Stable reusable tools

These scripts have clear reusable I/O or helper value and continuing engineering utility, even when they are not part of the current DEV-15 mainline.

## Stage 0 relevance

- `src/stage0_relevance/auto_tag_plga_gemini.py` — Gemini-based Zotero auto-tagging; reusable relevance-stage tool; reflected in `docs/tool_index.md`: yes.
- `src/stage0_relevance/auto_tag_plga_openai.py` — OpenAI-based Zotero auto-tagging; reusable relevance-stage tool; reflected in `docs/tool_index.md`: yes.
- `src/stage0_relevance/classify_gemini_grouped.py` — grouped LLM relevance classifier; reusable relevance-stage tool; reflected in `docs/tool_index.md`: yes.
- `src/stage0_relevance/fill_missing_snapshots.py` — snapshot backfill utility; reusable dataset-maintenance tool; reflected in `docs/tool_index.md`: yes.
- `src/stage0_relevance/prefilter_regex.py` — deterministic metadata prefilter; reusable relevance-stage tool; reflected in `docs/tool_index.md`: yes.
- `src/stage0_relevance/zotero_api_sync_selected.py` — Zotero sync/update utility; reusable raw-index maintenance tool; reflected in `docs/tool_index.md`: yes.
- `src/stage0_relevance/zotero_fetch_llm_relevant_pdfs.py` — fetches relevant PDFs/HTML; reusable relevance-stage tool; reflected in `docs/tool_index.md`: yes.
- `src/stage0_relevance/zotero_llm_relevant_interactive.py` — interactive relevance review helper; reusable reviewer tool; reflected in `docs/tool_index.md`: yes.
- `src/stage0_relevance/zotero_tag_sync.py` — Zotero tag sync tool; reusable relevance-stage tool; reflected in `docs/tool_index.md`: yes.

## Stage 1 cleaning

- `src/stage1_cleaning/clean_manifest_to_text.py` — cleaned-text generator from manifest; reusable Stage1 cleaning tool; reflected in `docs/tool_index.md`: yes.
- `src/stage1_cleaning/extract_tables_for_keys_v1.py` — per-key table extraction worker; reusable because `run_tables_extraction_for_dataset_v1.py` calls it directly; reflected in `docs/tool_index.md`: no.
- `src/stage1_cleaning/pdf2clean.py` — PDF cleaner with compatibility shim; reusable Stage1 cleaning tool; reflected in `docs/tool_index.md`: yes.
- `src/stage1_cleaning/run_tables_extraction_for_dataset_v1.py` — dataset-level table extraction runner; reusable Stage1 support tool; reflected in `docs/tool_index.md`: no, but named `ACTIVE_SUPPORTING` in `project/ACTIVE_PIPELINE_RUNBOOK.md`.
- `src/stage1_cleaning/zotero_raw_to_manifest.py` — raw-to-manifest builder; reusable Stage1 support tool; reflected in `docs/tool_index.md`: yes.

## Stage 2 sampling / extraction support

- `src/stage2_sampling_labels/build_evidence_bundle_for_keys_v1.py` — deterministic evidence-bundle builder from cleaned artifacts; reusable formulation-audit helper; reflected in `docs/tool_index.md`: no, but named active in `project/PIPELINE_SCRIPT_MAP.md`.
- `src/stage2_sampling_labels/build_key2txt_from_sample_manifest.py` — sample-local key-to-text builder; reusable Stage2 support tool; reflected in `docs/tool_index.md`: yes.
- `src/stage2_sampling_labels/build_strata_tags_from_text.py` — strata-tag generation from text; reusable sampling prep tool; reflected in `docs/tool_index.md`: yes.
- `src/stage2_sampling_labels/export_blockpack_audit_v7pilot_r3_fixparse.py` — deterministic packed-block audit exporter; reusable current-packer debug tool; reflected in `docs/tool_index.md`: no, but named `ACTIVE_SUPPORTING` in `project/ACTIVE_PIPELINE_RUNBOOK.md`.
- `src/stage2_sampling_labels/export_evidence_bundle_audit_xlsx_v1.py` — audit XLSX exporter for evidence bundles; reusable review aid with clear output contract; reflected in `docs/tool_index.md`: no.
- `src/stage2_sampling_labels/sample10_from_zotero_manifest.py` — historical sample10 builder still documented as active/supporting utility; reusable sample builder; reflected in `docs/tool_index.md`: yes.
- `src/stage2_sampling_labels/sample_from_manifest_html_first.py` — current split/sample builder; reusable Stage2 support tool; reflected in `docs/tool_index.md`: yes.

## Stage 3 GT / review helpers

- `src/stage3_gt/build_dev15_formulation_skeleton_review_v1.py` — DEV15 review-workbook scaffold builder; reusable GT-review utility; reflected in `docs/tool_index.md`: no.
- `src/stage3_gt/build_gt_template_from_conflict_queue.py` — conflict-to-GT template builder; reusable GT helper; reflected in `docs/tool_index.md`: yes.
- `src/stage3_gt/export_dev15_formulation_skeleton_gt_v1.py` — exports reviewed skeleton workbook to TSV; reusable GT utility; reflected in `docs/tool_index.md`: no.
- `src/stage3_gt/export_gt_annotation_view.py` — annotation-view XLSX exporter; reusable GT helper; reflected in `docs/tool_index.md`: yes.
- `src/stage3_gt/formulation_skeleton_common.py` — shared helpers for DEV15 skeleton review/export/validation; reusable support module; reflected in `docs/tool_index.md`: no.
- `src/stage3_gt/gt_summary_report.py` — GT-decision summary/report generator; reusable GT helper; reflected in `docs/tool_index.md`: yes.
- `src/stage3_gt/merge_gt_from_annotation_view.py` — merges human annotation XLSX back into authoritative TSV; reusable GT helper; reflected in `docs/tool_index.md`: yes.
- `src/stage3_gt/validate_dev15_formulation_skeleton_review_v1.py` — validates the DEV15 review workbook; reusable GT utility; reflected in `docs/tool_index.md`: no.

## Stage 4 evaluation / diagnostics tools

- `src/stage4_eval/apply_extracted_ee_dedup_v1.py` — deterministic dedup pass for extracted rows; reusable diagnostic/eval tool; reflected in `docs/tool_index.md`: yes.
- `src/stage4_eval/apply_formulation_grouping_v1.py` — formulation-grouping utility; reusable deterministic grouping tool even though not mainline default; reflected in `docs/tool_index.md`: yes.
- `src/stage4_eval/apply_global_baseline_inheritance_and_rerun_alignment_v1.py` — deterministic inheritance/alignment utility; reusable diagnostic/eval tool; reflected in `docs/tool_index.md`: yes.
- `src/stage4_eval/audit_top3_doi_root_cause_v1.py` — targeted root-cause audit builder for benchmark cases; reusable diagnosis tool; reflected in `docs/tool_index.md`: yes.
- `src/stage4_eval/build_boundary_alignment_diagnostics_pack_v1.py` — formulation-boundary diagnostics pack builder; reusable audit tool named active in script map; reflected in `docs/tool_index.md`: no.
- `src/stage4_eval/build_failure_profile_v1.py` — per-DOI failure profiling; reusable diagnostic tool; reflected in `docs/tool_index.md`: yes.
- `src/stage4_eval/build_goren_overlap_scaffold_v1.py` — benchmark overlap scaffold builder; reusable benchmark prep tool; reflected in `docs/tool_index.md`: yes.
- `src/stage4_eval/build_per_doi_diagnostics_v1.py` — per-DOI diagnostics pack builder; reusable diagnostic tool; reflected in `docs/tool_index.md`: yes.
- `src/stage4_eval/compare_drugname_sets_v1.py` — deterministic drug-name set comparison utility; reusable benchmark diagnostic tool; reflected in `docs/tool_index.md`: yes.
- `src/stage4_eval/compute_alignment_sensitivity_v1.py` — sensitivity-analysis helper for alignment; reusable evaluation tool; reflected in `docs/tool_index.md`: yes.
- `src/stage4_eval/compute_formulation_alignment_v1.py` — deterministic formulation alignment calculator; reusable evaluation tool named active in script map; reflected in `docs/tool_index.md`: yes.
- `src/stage4_eval/compute_goren_metrics_tables_v1.py` — benchmark metrics table builder; reusable evaluation tool; reflected in `docs/tool_index.md`: yes.
- `src/stage4_eval/compute_set_level_ee_match_v1.py` — set-level EE matching utility; reusable evaluation tool; reflected in `docs/tool_index.md`: yes.
- `src/stage4_eval/export_dev15_formulation_view_xlsx_v1.py` — human-auditable formulation view exporter; reusable review/audit tool named active in script map; reflected in `docs/tool_index.md`: no.
- `src/stage4_eval/precision_recovery_experiment_v1.py` — structured precision-recovery sweep with clear I/O; reusable evaluation experiment tool; reflected in `docs/tool_index.md`: yes.
- `src/stage4_eval/run_alignment_v3_surfactant_drugnorm.py` — deterministic normalized alignment runner; reusable benchmark/eval tool; reflected in `docs/tool_index.md`: yes.
## Stage 5 benchmark / schema / export tools

- `src/stage5_benchmark/analyze_row_membership_core_v1.py` — core-level row-membership analyzer; reusable benchmark tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/analyze_row_membership_v1.py` — row-membership analyzer; reusable benchmark tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/build_goren_overlap_manifest.py` — overlap-manifest builder; reusable benchmark prep tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/build_two_table_schema_v1.py` — schema_v1 two-table builder; reusable schema tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/build_two_table_schema_v2.py` — schema_v2 builder; reusable schema tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/build_two_table_schema_v3.py` — schema_v3 builder; reusable schema tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/copy_goren_dataset.py` — curated benchmark copier; reusable benchmark prep tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/debug_doi_overlap.py` — DOI overlap inspection utility; reusable benchmark debug tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/derive_doe_coded_factors_v1.py` — deterministic DOE coded-factor derivation tool; reusable benchmark/schema tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/evaluate_against_goren.py` — benchmark evaluator; reusable benchmark tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/export_audit_to_excel_v1.py` — audit workbook exporter; reusable review tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/export_full_database_v1.py` — stable full-database exporter; reusable publication/export tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/formulation_core_signature_v1.py` — core-signature logic module; reusable Stage5 support tool named in `project/ACTIVE_PIPELINE_RUNBOOK.md`; reflected in `docs/tool_index.md`: no.
- `src/stage5_benchmark/inspect_schemas.py` — schema inspection utility; reusable benchmark/schema tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/make_audit_pack_v1.py` — deterministic parsing/derivation audit-pack generator; reusable benchmark audit tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/report_schema_v2_v3_core_diff_v1.py` — core-diff reporter across schema versions; reusable benchmark comparison tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/run_alignment_eval_core_v1.py` — core-level alignment runner; reusable benchmark tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/run_alignment_eval_schema_v3_v1.py` — schema-v3 alignment runner; reusable benchmark comparison tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/run_alignment_eval_v1.py` — alignment runner; reusable benchmark tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/run_core_eval_pipeline_v1.py` — orchestrates core-eval pipeline; reusable benchmark workflow tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/run_derivation_v1.py` — derivation layer runner; reusable benchmark tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/run_evidence_realign_v1.py` — deterministic evidence realignment utility; reusable benchmark hardening tool; reflected in `docs/tool_index.md`: no, but referenced by `docs/methods/evidence_alignment_hardening_plan.md`.
- `src/stage5_benchmark/run_evidence_token_qc_v1.py` — evidence-token QC runner; reusable benchmark QC tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/run_formulation_core_signature_v1.py` — runner for core-signature module; reusable benchmark/schema tool; reflected in `docs/tool_index.md`: no.
- `src/stage5_benchmark/run_projection_core_to_curated_v1.py` — projection from core schema to curated columns; reusable benchmark tool; reflected in `docs/tool_index.md`: yes.
- `src/stage5_benchmark/run_projection_to_curated_v1.py` — projection to curated schema; reusable benchmark tool; reflected in `docs/tool_index.md`: yes.

## Stage 5 merge/publish

- `src/stage5_merge_publish/merge_results.py` — final verified-results merger; reusable final export tool and active in script map; reflected in `docs/tool_index.md`: yes.

## Shared utils

- `src/utils/build_dataset_root_from_legacy_v1.py` — dataset-root migration helper from legacy layouts; reusable utility with clear I/O; reflected in `docs/tool_index.md`: no.
- `src/utils/build_dataset_split_dev_v1.py` — DEV split builder; reusable utility; reflected in `docs/tool_index.md`: no.
- `src/utils/build_dataset_split_test_v1.py` — TEST split builder with DEV exclusion; reusable utility; reflected in `docs/tool_index.md`: no.
- `src/utils/build_ee_only_manifest.py` — EE-only manifest builder; reusable utility; reflected in `docs/tool_index.md`: yes.
- `src/utils/build_global_zotero_index_v1.py` — global Zotero index builder; reusable utility; reflected in `docs/tool_index.md`: no.
- `src/utils/convert_sample_manifest.py` — manifest conversion helper; reusable utility; reflected in `docs/tool_index.md`: yes.
- `src/utils/convert_sample_manifest_to_tsv.py` — manifest-to-TSV conversion helper; reusable utility; reflected in `docs/tool_index.md`: yes.
- `src/utils/gemini_models.py` — Gemini model helper definitions; reusable support module; reflected in `docs/tool_index.md`: yes.
- `src/utils/html_parser.py` — shared HTML parsing utility; reusable support module; reflected in `docs/tool_index.md`: yes.
- `src/utils/model_policy.py` — shared model-policy validator/constants imported by multiple extractors and preflight; reusable support module; reflected in `docs/tool_index.md`: no.
- `src/utils/paths.py` — canonical repository path resolver; reusable core utility and active supporting script in runbook; reflected in `docs/tool_index.md`: yes.
- `src/utils/run_id.py` — run-id generation/validation helper; reusable core utility; reflected in `docs/tool_index.md`: yes.
- `src/utils/run_latest.py` — latest-run pointer and fingerprint helper imported by multiple scripts; reusable support module; reflected in `docs/tool_index.md`: no.
- `src/utils/run_preflight.py` — run reuse/new-run preflight helper; reusable core utility and active supporting script in runbook; reflected in `docs/tool_index.md`: no.
- `src/utils/scan_ee_coverage.py` — EE coverage scanner; reusable utility; reflected in `docs/tool_index.md`: yes.
- `src/utils/split_registry_v1.py` — DEV split registry loader used by dataset split/layout validators; reusable utility; reflected in `docs/tool_index.md`: no.
- `src/utils/test_gemini.py` — lightweight Gemini connectivity/probe helper; reusable engineering utility; reflected in `docs/tool_index.md`: yes.
- `src/utils/validate_dataset_layout_v1.py` — dataset layout validator; reusable utility; reflected in `docs/tool_index.md`: no.

# 4. Archived methods / historical baselines

These scripts retain historical or comparative value, but they are not the current mainline.

## Legacy subtree

- `src/legacy/20260130/csv2clean_manifest.py` — historical manifest/cleaning baseline; not current because Stage1 uses `zotero_raw_to_manifest.py` and `clean_manifest_to_text.py`; should later be moved under a dedicated archive subtree: already under `src/legacy/`.
- `src/legacy/20260130/pdf2clean.py` — historical PDF cleaning baseline; not current because Stage1 uses `src/stage1_cleaning/pdf2clean.py`; should later be moved under a dedicated archive subtree: already under `src/legacy/`.
- `src/legacy/20260130/zotero_csv_to_manifest_tsv.py` — historical Zotero-to-manifest path; not current because manifest building has newer Stage1 utilities; should later be moved under a dedicated archive subtree: already under `src/legacy/`.
- `src/legacy/20260131/clean_manifest_to_text.py` — historical cleaned-text baseline; not current because the active runbook names `src/stage1_cleaning/clean_manifest_to_text.py`; should later be moved under a dedicated archive subtree: already under `src/legacy/`.
- `src/legacy/20260131/sample_from_manifest_html_first.py` — historical sample builder baseline; not current because the active runbook names `src/stage2_sampling_labels/sample_from_manifest_html_first.py`; should later be moved under a dedicated archive subtree: already under `src/legacy/`.
- `src/legacy/20260202/gt_tool.py` — older manual GT tool; not current because Stage3 GT maintenance now uses export/merge/report utilities; should later be moved under a dedicated archive subtree: already under `src/legacy/`.
- `src/legacy/20260202/gt_tool_v3.py` — later legacy GT tool variant; not current because Stage3 GT maintenance now uses export/merge/report utilities; should later be moved under a dedicated archive subtree: already under `src/legacy/`.

## Older weak-label / pilot extractor families

- `src/stage1_cleaning/find_html_table_candidates_v1.py` — exploratory HTML-table candidate picker for pilot expansion; not current because no active governance doc names it and table extraction now centers on dataset runners; should later be moved under a dedicated archive subtree: yes.
- `src/stage2_sampling_labels/auto_extract_weak_labels.py` — older generic extraction entry; not current because the active runbook explicitly says not to default to it for current DEV work; should later be moved under a dedicated archive subtree: yes.
- `src/stage2_sampling_labels/auto_extract_weak_labels_v3.py` — weak-label logic v3 baseline; not current because script map marks it legacy and runbook excludes it from current DEV work; should later be moved under a dedicated archive subtree: yes.
- `src/stage2_sampling_labels/auto_extract_weak_labels_v4.py` — older extraction baseline retained for comparison; not current because the active extractor is `v7pilot_r3_fixparse`; should later be moved under a dedicated archive subtree: yes.
- `src/stage2_sampling_labels/auto_extract_weak_labels_v5.py` — older extractor baseline; not current because runbook excludes it from current DEV work; should later be moved under a dedicated archive subtree: yes.
- `src/stage2_sampling_labels/auto_extract_weak_labels_v5_G3.py` — dual-model-era or alternate-model historical extractor path; not current because runbook excludes it from current DEV work; should later be moved under a dedicated archive subtree: yes.
- `src/stage2_sampling_labels/auto_extract_weak_labels_v6.py` — prior mainline semantic extraction baseline; not current because current DEV work uses `v7pilot_r3_fixparse`; should later be moved under a dedicated archive subtree: yes.
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot.py` — early v7 pilot extractor; not current because current DEV work uses the later `r3_fixparse` variant; should later be moved under a dedicated archive subtree: yes.
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r2.py` — intermediate pilot iteration; not current because current DEV work uses `r3_fixparse`; should later be moved under a dedicated archive subtree: yes.
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3.py` — pilot r3 baseline before fix variants; not current because current DEV work uses `r3_fixparse`; should later be moved under a dedicated archive subtree: yes.
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixflat.py` — comparative pilot variant for flattening bugfix testing; not current because active flow uses `r3_fixparse`; should later be moved under a dedicated archive subtree: yes.
## Stage 4 comparative, dual-model, and pilot-debug methods

- `src/stage4_eval/audit_doi_raw_field_mapping_v7pilot.py` — v7pilot raw-field mapping audit for targeted papers; not current because it is an unregistered pilot-debug path rather than the active evaluator; should later be moved under a dedicated archive subtree: yes.
- `src/stage4_eval/auto_extract_multimodel.py` — multimodel extraction path from earlier comparative workflows; not current because runbook explicitly excludes multimodel scripts as default entrypoints; should later be moved under a dedicated archive subtree: yes.
- `src/stage4_eval/build_doi_value_flow_audit_v7pilot.py` — v7pilot DOI value-flow audit builder; not current because it is pilot-specific debug support, not part of active DEV counting; should later be moved under a dedicated archive subtree: yes.
- `src/stage4_eval/build_v7pilot_field_mapping_audit.py` — v7pilot field-mapping audit builder; not current because it supports a comparative pilot-debug thread rather than the active flow; should later be moved under a dedicated archive subtree: yes.
- `src/stage4_eval/compare_v7pilot_scope_r1_r2.py` — result comparison between pilot scopes/iterations; not current because it exists only to compare historical pilot paths; should later be moved under a dedicated archive subtree: yes.
- `src/stage4_eval/export_dev15_dashboard_and_audit_pack_v1.py` — one-off dashboard/audit pack exporter for manual review; not current because active reviewer output is `build_dev15_review_workbook_v1.py`; should later be moved under a dedicated archive subtree: yes.
- `src/stage4_eval/inspect_formulation_view_inventory_v1.py` — formulation-view inventory/debug inspector; not current because it is an unregistered debug support script; should later be moved under a dedicated archive subtree: yes.
- `src/stage4_eval/multi_model_consensus_vote.py` — conservative two-model consensus weak-label path; not current because current mainline is single-path formulation-instance extraction/eval; should later be moved under a dedicated archive subtree: yes.
- `src/stage4_eval/multi_model_extract_tier1.py` — historical two-model extraction tier; not current because runbook excludes multimodel scripts from current DEV work; should later be moved under a dedicated archive subtree: yes.
- `src/stage4_eval/multi_model_extract_tier2.py` — historical two-model extraction tier; not current because runbook excludes multimodel scripts from current DEV work; should later be moved under a dedicated archive subtree: yes.
- `src/stage4_eval/multi_model_merge_qc.py` — historical two-model merge/QC comparator; not current because current DEV work does not use the multimodel path; should later be moved under a dedicated archive subtree: yes.
- `src/stage4_eval/test_doe_coordinate_reconciliation_v1.py` — experimental isolated validation script for DoE reconciliation; not current because runbook explicitly marks it experimental and says validated logic is already in `eval_weak_labels_v7pilot3.py`; should later be moved under a dedicated archive subtree: yes.

## Stage 5 historical benchmark/debug methods

- `src/stage5_benchmark/audit_evidence_resolver_v1.py` — internal resolver used by a historical audit-pack path; not current because no active governance doc names it and it feeds only archived audit workflows; should later be moved under a dedicated archive subtree: yes.
- `src/stage5_benchmark/build_audit_pack_human_evidence_v1.py` — human evidence audit pack builder for branch-specific benchmark review; not current because it is not part of the active formulation-instance mainline; should later be moved under a dedicated archive subtree: yes.
- `src/stage5_benchmark/build_doe_signature_injection_audit_pack_v1.py` — DOE signature injection audit builder for benchmark experimentation; not current because it supports a narrow experimental audit path; should later be moved under a dedicated archive subtree: yes.
- `src/stage5_benchmark/build_signature_v1_audit_pack.py` — historical signature audit-pack generator; not current because it supports a narrow audit workflow rather than the active path; should later be moved under a dedicated archive subtree: yes.
- `src/stage5_benchmark/check_baseline_size_before_freeze_drying_v1.py` — one-paper regression check for a single derived field; not current because it is a targeted regression probe, not a reusable mainline stage; should later be moved under a dedicated archive subtree: yes.
- `src/stage5_benchmark/check_explicit_formulation_id_core_split_v2.py` — narrow schema/core split check; not current because it is a focused validation probe for one comparative method; should later be moved under a dedicated archive subtree: yes.
- `src/stage5_benchmark/debug_paper_local_tables_registry_v1.py` — table-registry debug script tied to extraction-table auditing; not current because it is a debug helper invoked from extraction checks, not a mainline stage; should later be moved under a dedicated archive subtree: yes.
- `src/stage5_benchmark/export_human_optimization_audit_10_v1.py` — reviewer workbook exporter for a narrow optimization audit set; not current because it is a branch-specific debug/review method; should later be moved under a dedicated archive subtree: yes.
- `src/stage5_benchmark/run_audit10_regression_check_v1.py` — regression checker for the audit10 reviewer workflow; not current because it validates a narrow historical audit process; should later be moved under a dedicated archive subtree: yes.

# 5. Delete candidates

No scripts are classified as `DELETE_CANDIDATE` in this pass.

Reason:

- Every non-mainline script in `src/` still has at least one of:
  - explicit governance evidence,
  - inclusion in `docs/tool_index.md`,
  - clear import/call relationships,
  - or meaningful historical/comparative provenance.

Conservative conclusion:

- `DELETE_CANDIDATE`: none

# 6. Explicit review of old comparative/GT paths

## Dual-model extraction paths

- `src/stage4_eval/multi_model_extract_tier1.py` — `ARCHIVED_METHOD`; historical dual-model extraction tier, not named as active and explicitly non-default in the runbook.
- `src/stage4_eval/multi_model_extract_tier2.py` — `ARCHIVED_METHOD`; same reason as tier1.
- `src/stage4_eval/multi_model_merge_qc.py` — `ARCHIVED_METHOD`; result-comparison/merge path for the dual-model workflow, not current mainline.
- `src/stage4_eval/multi_model_consensus_vote.py` — `ARCHIVED_METHOD`; conservative two-model consensus pipeline, valuable historical method but not the current architecture center.
- `src/stage4_eval/auto_extract_multimodel.py` — `ARCHIVED_METHOD`; multimodel extraction runner retained for provenance/comparison only.
- `src/stage2_sampling_labels/auto_extract_weak_labels_v5_G3.py` — `ARCHIVED_METHOD`; older alternative-model baseline, informative historically but not current mainline.

## Result-comparison paths

- `src/stage4_eval/compare_v7pilot_scope_r1_r2.py` — `ARCHIVED_METHOD`; direct comparison between two pilot scopes, not a continuing mainline tool.
- `src/stage4_eval/compute_alignment_sensitivity_v1.py` — `STABLE_TOOL`; comparison-style evaluation utility with reusable I/O, already reflected in `docs/tool_index.md`.
- `src/stage4_eval/compare_drugname_sets_v1.py` — `STABLE_TOOL`; deterministic comparison tool with clear continuing utility.
- `src/stage5_benchmark/report_schema_v2_v3_core_diff_v1.py` — `STABLE_TOOL`; schema comparison tool with ongoing utility.

## Manual GT / human review / arbitration helper paths

- `src/stage3_gt/build_gt_template_from_conflict_queue.py` — `STABLE_TOOL`; current manual GT helper path with clear input/output.
- `src/stage3_gt/export_gt_annotation_view.py` — `STABLE_TOOL`; current annotation-view helper.
- `src/stage3_gt/merge_gt_from_annotation_view.py` — `STABLE_TOOL`; current merge-back helper.
- `src/stage3_gt/gt_summary_report.py` — `STABLE_TOOL`; current GT reporting helper.
- `src/stage3_gt/build_dev15_formulation_skeleton_review_v1.py` — `STABLE_TOOL`; current DEV15 skeleton review helper, even though not part of the active extraction/eval mainline.
- `src/stage3_gt/validate_dev15_formulation_skeleton_review_v1.py` — `STABLE_TOOL`; validation helper for the above.
- `src/stage3_gt/export_dev15_formulation_skeleton_gt_v1.py` — `STABLE_TOOL`; export helper for the above.
- `src/legacy/20260202/gt_tool.py` — `ARCHIVED_METHOD`; older GT path retained for provenance only.
- `src/legacy/20260202/gt_tool_v3.py` — `ARCHIVED_METHOD`; later legacy GT path retained for provenance only.
- `src/stage4_eval/build_dev15_review_workbook_v1.py` — `ACTIVE_ENTRYPOINT`; active reviewer workbook generator in the current flow.
- `src/stage2_sampling_labels/export_blockpack_audit_v7pilot_r3_fixparse.py` — `STABLE_TOOL`; current audit helper for the active packer.
- `src/stage4_eval/export_dev15_formulation_view_xlsx_v1.py` — `STABLE_TOOL`; reusable human-review exporter.
- `src/stage4_eval/export_dev15_dashboard_and_audit_pack_v1.py` — `ARCHIVED_METHOD`; review-oriented but not active and not clearly part of the current maintained tool stack.

## Older weak-label pilot / blockpack / tableheavy / fixflat paths

- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot.py` — `ARCHIVED_METHOD`; early pilot baseline.
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r2.py` — `ARCHIVED_METHOD`; intermediate pilot baseline.
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3.py` — `ARCHIVED_METHOD`; pre-fix pilot baseline.
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixflat.py` — `ARCHIVED_METHOD`; comparative fixflat variant.
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py` — `ACTIVE_ENTRYPOINT`; current mainline pilot extractor.
- `src/stage2_sampling_labels/export_blockpack_audit_v7pilot_r3_fixparse.py` — `STABLE_TOOL`; current active-packer audit helper.
- `src/stage4_eval/audit_doi_raw_field_mapping_v7pilot.py` — `ARCHIVED_METHOD`; pilot-debug audit path.
- `src/stage4_eval/build_doi_value_flow_audit_v7pilot.py` — `ARCHIVED_METHOD`; pilot-debug audit path.
- `src/stage4_eval/build_v7pilot_field_mapping_audit.py` — `ARCHIVED_METHOD`; pilot-debug audit path.
# 7. Proposed next-step reduction plan

Do not execute in this pass.

## Scripts that should remain in stage directories

- Keep all `ACTIVE_ENTRYPOINT` scripts where they are.
- Keep current reusable stage support and utility scripts in place:
  - Stage0 relevance scripts
  - Stage1 cleaned-content / table-extraction scripts
  - Stage3 GT helper scripts
  - maintained Stage4/Stage5 benchmark utilities
  - shared `src/utils/` support modules

## Scripts that should be marked archived

- Mark all `src/legacy/**` scripts as explicitly archived in future inventory docs.
- Mark older extractor variants under `src/stage2_sampling_labels/auto_extract_weak_labels*` except `v7pilot_r3_fixparse.py` as archived historical methods.
- Mark Stage4 multimodel/comparative paths as archived historical methods.
- Mark narrow Stage5 branch-specific audit/regression/debug scripts as archived historical methods.

## Strongest reduction candidates for a later human-reviewed pass

No direct delete actions are proposed now.

If a later reduction pass is required, the strongest review targets are:

- narrow one-off Stage4 pilot-debug scripts:
  - `audit_doi_raw_field_mapping_v7pilot.py`
  - `build_doi_value_flow_audit_v7pilot.py`
  - `build_v7pilot_field_mapping_audit.py`
  - `compare_v7pilot_scope_r1_r2.py`
- narrow Stage5 branch-specific audit/regression helpers:
  - `build_audit_pack_human_evidence_v1.py`
  - `build_doe_signature_injection_audit_pack_v1.py`
  - `build_signature_v1_audit_pack.py`
  - `export_human_optimization_audit_10_v1.py`
  - `run_audit10_regression_check_v1.py`
  - `check_baseline_size_before_freeze_drying_v1.py`

These are still better treated as archival candidates rather than deletion candidates until provenance needs are reviewed by a human.

# 8. Summary counts

- `ACTIVE_ENTRYPOINT`: 3
- `STABLE_TOOL`: 90
- `ARCHIVED_METHOD`: 39
- `DELETE_CANDIDATE`: 0
- uncertain items: 0
