**Scripts That Should Remain In `src/` As Current Active Assets**

- `src/stage1_cleaning/clean_manifest_to_text.py`: mainline_active | Document cleaning and structural normalization | Generate cleaned text assets from manifest rows
- `src/stage1_cleaning/extract_tables_for_keys_v1.py`: branch_active | Document cleaning and structural normalization | extract_tables_for_keys_v1 utility or helper
- `src/stage1_cleaning/find_html_table_candidates_v1.py`: branch_active | Document cleaning and structural normalization | find_html_table_candidates_v1 utility or helper
- `src/stage1_cleaning/pdf2clean.py`: branch_active | Document cleaning and structural normalization | pdf2clean utility or helper
- `src/stage1_cleaning/run_tables_extraction_for_dataset_v1.py`: mainline_active | Document cleaning and structural normalization | Generate dataset-local table assets
- `src/stage1_cleaning/zotero_raw_to_manifest.py`: mainline_active | Document cleaning and structural normalization | Build cleaned manifest from Zotero-derived raw inputs
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`: mainline_active | LLM candidate formulation extraction | Current Stage2 candidate formulation-instance extractor
- `src/stage2_sampling_labels/build_evidence_bundle_for_keys_v1.py`: branch_active | Evidence block construction and packing | Build structured artifact, audit pack, or manifest
- `src/stage2_sampling_labels/build_key2txt_from_sample_manifest.py`: mainline_active | Evidence block construction and packing | Build structured artifact, audit pack, or manifest
- `src/stage2_sampling_labels/build_strata_tags_from_text.py`: branch_active | Evidence block construction and packing | Build structured artifact, audit pack, or manifest
- `src/stage2_sampling_labels/sample10_from_zotero_manifest.py`: branch_active | LLM candidate formulation extraction | sample10_from_zotero_manifest utility or helper
- `src/stage2_sampling_labels/sample_from_manifest_html_first.py`: mainline_active | LLM candidate formulation extraction | sample_from_manifest_html_first utility or helper
- `src/stage4_eval/apply_extracted_ee_dedup_v1.py`: branch_active | Instance-level evaluation and diagnostics | Apply grouping, inheritance, or rewrite heuristics
- `src/stage4_eval/build_dev15_review_workbook_v1.py`: branch_active | Instance-level evaluation and diagnostics | Build DEV-15 reviewer workbook from evaluation outputs
- `src/stage4_eval/compute_formulation_alignment_v1.py`: branch_active | Instance-level evaluation and diagnostics | Compute analysis or evaluation summary tables
- `src/stage4_eval/compute_set_level_ee_match_v1.py`: branch_active | Instance-level evaluation and diagnostics | Compute analysis or evaluation summary tables
- `src/stage4_eval/eval_weak_labels_v7pilot3.py`: mainline_active | Instance-level evaluation and diagnostics | Current Stage4 candidate-instance evaluator
- `src/stage4_eval/inspect_formulation_view_inventory_v1.py`: branch_active | Instance-level evaluation and diagnostics | Inspect or debug repository artifacts
- `src/stage5_benchmark/formulation_core_signature_v1.py`: mainline_active | Light guardrail / normalization / benchmark export | Build or run formulation core-signature logic
- `src/stage5_merge_publish/merge_results.py`: mainline_active | Final formulation table and modeling export | Merge validated formulation outputs into publishable tables
- `src/utils/build_dataset_root_from_legacy_v1.py`: branch_active | Cross-cutting support utility | Build structured artifact, audit pack, or manifest
- `src/utils/build_dataset_split_dev_v1.py`: branch_active | Cross-cutting support utility | Build structured artifact, audit pack, or manifest
- `src/utils/build_dataset_split_test_v1.py`: branch_active | Cross-cutting support utility | Build structured artifact, audit pack, or manifest
- `src/utils/build_ee_only_manifest.py`: branch_active | Cross-cutting support utility | Build structured artifact, audit pack, or manifest
- `src/utils/build_global_zotero_index_v1.py`: branch_active | Cross-cutting support utility | Build structured artifact, audit pack, or manifest
- `src/utils/convert_sample_manifest.py`: branch_active | Cross-cutting support utility | convert_sample_manifest utility or helper
- `src/utils/convert_sample_manifest_to_tsv.py`: branch_active | Cross-cutting support utility | convert_sample_manifest_to_tsv utility or helper
- `src/utils/gemini_models.py`: branch_active | Cross-cutting support utility | gemini_models utility or helper
- `src/utils/html_parser.py`: branch_active | Cross-cutting support utility | html_parser utility or helper
- `src/utils/model_policy.py`: branch_active | Cross-cutting support utility | model_policy utility or helper
- `src/utils/paths.py`: mainline_active | Cross-cutting support utility | paths utility or helper
- `src/utils/run_id.py`: branch_active | Cross-cutting support utility | Run orchestrated utility or evaluation step
- `src/utils/run_latest.py`: branch_active | Cross-cutting support utility | Run orchestrated utility or evaluation step
- `src/utils/run_preflight.py`: mainline_active | Cross-cutting support utility | Run orchestrated utility or evaluation step
- `src/utils/scan_ee_coverage.py`: branch_active | Cross-cutting support utility | scan_ee_coverage utility or helper
- `src/utils/split_registry_v1.py`: branch_active | Cross-cutting support utility | split_registry_v1 utility or helper
- `src/utils/validate_dataset_layout_v1.py`: branch_active | Cross-cutting support utility | validate_dataset_layout_v1 utility or helper

**Scripts That Should Remain But Be Explicitly Marked Branch-Only**

- `src/stage0_relevance/auto_tag_plga_gemini.py`: branch_active | Corpus and document assets | auto_tag_plga_gemini utility or helper
- `src/stage0_relevance/auto_tag_plga_openai.py`: branch_active | Corpus and document assets | auto_tag_plga_openai utility or helper
- `src/stage0_relevance/classify_gemini_grouped.py`: branch_active | Corpus and document assets | classify_gemini_grouped utility or helper
- `src/stage0_relevance/fill_missing_snapshots.py`: branch_active | Corpus and document assets | fill_missing_snapshots utility or helper
- `src/stage0_relevance/prefilter_regex.py`: branch_active | Corpus and document assets | prefilter_regex utility or helper
- `src/stage0_relevance/zotero_api_sync_selected.py`: branch_active | Corpus and document assets | zotero_api_sync_selected utility or helper
- `src/stage0_relevance/zotero_fetch_llm_relevant_pdfs.py`: branch_active | Corpus and document assets | zotero_fetch_llm_relevant_pdfs utility or helper
- `src/stage0_relevance/zotero_llm_relevant_interactive.py`: branch_active | Corpus and document assets | zotero_llm_relevant_interactive utility or helper
- `src/stage0_relevance/zotero_tag_sync.py`: branch_active | Corpus and document assets | zotero_tag_sync utility or helper
- `src/stage2_sampling_labels/diagnose_5gif3d8w_axis_applicability_v1.py`: branch_active | LLM candidate formulation extraction | Paper-specific diagnostic analysis helper
- `src/stage2_sampling_labels/diagnose_5gif3d8w_root_cause_v1.py`: branch_active | LLM candidate formulation extraction | Paper-specific diagnostic analysis helper
- `src/stage2_sampling_labels/export_blockpack_audit_v7pilot_r3_fixparse.py`: branch_active | Evidence block construction and packing | Export workbook, TSV, or audit artifact
- `src/stage2_sampling_labels/export_evidence_bundle_audit_xlsx_v1.py`: branch_active | LLM candidate formulation extraction | Export workbook, TSV, or audit artifact
- `src/stage2_sampling_labels/run_targeted_stage2_regression_v1.py`: branch_active | LLM candidate formulation extraction | Run targeted Stage2 regression set with documented context
- `src/stage4_eval/audit_top3_doi_root_cause_v1.py`: branch_active | Instance-level evaluation and diagnostics | audit_top3_doi_root_cause_v1 utility or helper
- `src/stage4_eval/precision_recovery_experiment_v1.py`: branch_active | Instance-level evaluation and diagnostics | precision_recovery_experiment_v1 utility or helper
- `src/stage5_benchmark/audit_evidence_resolver_v1.py`: branch_active | Light guardrail / normalization / benchmark export | audit_evidence_resolver_v1 utility or helper
- `src/stage5_benchmark/build_audit_pack_human_evidence_v1.py`: branch_active | Light guardrail / normalization / benchmark export | Build structured artifact, audit pack, or manifest
- `src/stage5_benchmark/build_doe_signature_injection_audit_pack_v1.py`: branch_active | Light guardrail / normalization / benchmark export | Build structured artifact, audit pack, or manifest
- `src/stage5_benchmark/build_signature_v1_audit_pack.py`: branch_active | Light guardrail / normalization / benchmark export | Build structured artifact, audit pack, or manifest
- `src/stage5_benchmark/build_two_table_schema_v1.py`: branch_active | Light guardrail / normalization / benchmark export | Build normalized two-table benchmark or export schema
- `src/stage5_benchmark/build_two_table_schema_v2.py`: branch_active | Light guardrail / normalization / benchmark export | Build normalized two-table benchmark or export schema
- `src/stage5_benchmark/build_two_table_schema_v3.py`: branch_active | Light guardrail / normalization / benchmark export | Build normalized two-table benchmark or export schema
- `src/stage5_benchmark/check_baseline_size_before_freeze_drying_v1.py`: branch_active | Light guardrail / normalization / benchmark export | check_baseline_size_before_freeze_drying_v1 utility or helper
- `src/stage5_benchmark/check_explicit_formulation_id_core_split_v2.py`: branch_active | Light guardrail / normalization / benchmark export | check_explicit_formulation_id_core_split_v2 utility or helper
- `src/stage5_benchmark/debug_paper_local_tables_registry_v1.py`: branch_active | Light guardrail / normalization / benchmark export | Inspect or debug repository artifacts
- `src/stage5_benchmark/derive_doe_coded_factors_v1.py`: branch_active | Light guardrail / normalization / benchmark export | derive_doe_coded_factors_v1 utility or helper
- `src/stage5_benchmark/export_full_database_v1.py`: branch_active | Light guardrail / normalization / benchmark export | Export workbook, TSV, or audit artifact
- `src/stage5_benchmark/export_human_optimization_audit_10_v1.py`: branch_active | Light guardrail / normalization / benchmark export | Export workbook, TSV, or audit artifact
- `src/stage5_benchmark/inspect_schemas.py`: branch_active | Light guardrail / normalization / benchmark export | Inspect or debug repository artifacts
- `src/stage5_benchmark/run_alignment_eval_core_v1.py`: branch_active | Light guardrail / normalization / benchmark export | Run orchestrated utility or evaluation step
- `src/stage5_benchmark/run_alignment_eval_schema_v3_v1.py`: branch_active | Light guardrail / normalization / benchmark export | Run orchestrated utility or evaluation step
- `src/stage5_benchmark/run_alignment_eval_v1.py`: branch_active | Light guardrail / normalization / benchmark export | Run orchestrated utility or evaluation step
- `src/stage5_benchmark/run_audit10_regression_check_v1.py`: branch_active | Light guardrail / normalization / benchmark export | Run orchestrated utility or evaluation step
- `src/stage5_benchmark/run_core_eval_pipeline_v1.py`: branch_active | Light guardrail / normalization / benchmark export | Run orchestrated utility or evaluation step
- `src/stage5_benchmark/run_derivation_v1.py`: branch_active | Light guardrail / normalization / benchmark export | Run benchmark derivation from legacy weak-label rows
- `src/stage5_benchmark/run_evidence_realign_v1.py`: branch_active | Light guardrail / normalization / benchmark export | Run orchestrated utility or evaluation step
- `src/stage5_benchmark/run_evidence_token_qc_v1.py`: branch_active | Light guardrail / normalization / benchmark export | Run orchestrated utility or evaluation step
- `src/stage5_benchmark/run_formulation_core_signature_v1.py`: branch_active | Light guardrail / normalization / benchmark export | Run orchestrated utility or evaluation step
- `src/stage5_benchmark/run_projection_core_to_curated_v1.py`: branch_active | Light guardrail / normalization / benchmark export | Run orchestrated utility or evaluation step
- `src/stage5_benchmark/run_projection_to_curated_v1.py`: branch_active | Light guardrail / normalization / benchmark export | Run orchestrated utility or evaluation step

**Scripts That Should Be Moved To Archive**
- None remaining inside `src/` after cleanup wave 2.

**Scripts That Are Delete Candidates After Human Confirmation**
- None remaining inside `src/` after cleanup wave 2. Delete-candidate quarantine now lives outside `src/` under `archive/delete_candidates_pending_confirmation/`.

**Top 10 Most Problematic Scripts**

- `src/stage0_relevance/auto_tag_plga_gemini.py`: branch_active | Corpus and document assets | auto_tag_plga_gemini utility or helper
- `src/stage0_relevance/auto_tag_plga_openai.py`: branch_active | Corpus and document assets | auto_tag_plga_openai utility or helper
- `src/stage0_relevance/classify_gemini_grouped.py`: branch_active | Corpus and document assets | classify_gemini_grouped utility or helper
- `src/stage0_relevance/fill_missing_snapshots.py`: branch_active | Corpus and document assets | fill_missing_snapshots utility or helper
- `src/stage0_relevance/prefilter_regex.py`: branch_active | Corpus and document assets | prefilter_regex utility or helper
- `src/stage0_relevance/zotero_api_sync_selected.py`: branch_active | Corpus and document assets | zotero_api_sync_selected utility or helper
- `src/stage0_relevance/zotero_fetch_llm_relevant_pdfs.py`: branch_active | Corpus and document assets | zotero_fetch_llm_relevant_pdfs utility or helper
- `src/stage0_relevance/zotero_llm_relevant_interactive.py`: branch_active | Corpus and document assets | zotero_llm_relevant_interactive utility or helper
- `src/stage0_relevance/zotero_tag_sync.py`: branch_active | Corpus and document assets | zotero_tag_sync utility or helper
- `src/stage1_cleaning/pdf2clean.py`: branch_active | Document cleaning and structural normalization | pdf2clean utility or helper