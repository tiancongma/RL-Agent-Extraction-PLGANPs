# 1. Why the first-pass classification was too broad

The first pass overused `STABLE_TOOL` for scripts that were merely well-formed or documented.
That was too permissive for three reasons:

- `docs/tool_index.md` lists many scripts as reusable, but it is an inventory snapshot, not a maintenance guarantee.
- Many `*_v1.py` scripts have clear CLI arguments yet still encode a frozen benchmark, fixed run family, or one-off audit question.
- Several scripts previously called stable are actually narrow DEV15/Goren diagnostics, comparison helpers, or schema-iteration artifacts rather than maintained cross-task utilities.

Stricter rule applied in this pass:

- keep `STABLE_TOOL` only when the script has a clear reusable contract, a role a new contributor would recognize as maintained, and likely continuing value in the current or near-future pipeline;
- downgrade benchmark-specific diagnostics, frozen pilot helpers, and one-off audit builders to `ARCHIVED_METHOD`;
- use `DELETE_CANDIDATE` only where the script looks like residue rather than reusable infrastructure or meaningful historical method.

# 2. Confirmed ACTIVE_ENTRYPOINT

- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py` ！ current Stage2 default extractor; named `ACTIVE_MAINLINE` in `project/ACTIVE_PIPELINE_RUNBOOK.md` and used in `project/ACTIVE_PIPELINE_FLOW.md` steps `DEV15_01` and `DEV15_03`.
- `src/stage4_eval/eval_weak_labels_v7pilot3.py` ！ current Stage4 evaluator and reconciliation seam; named `ACTIVE_MAINLINE` in the runbook and used in flow steps `DEV15_02` and `DEV15_04`.
- `src/stage4_eval/build_dev15_review_workbook_v1.py` ！ active reviewer-workbook step; used in flow step `DEV15_05` and named `ACTIVE_SUPPORTING` in the runbook.

# 3. Confirmed STABLE_TOOL

Only scripts that still clear the stricter stability bar are kept here.

## Stage 0 relevance

- `src/stage0_relevance/auto_tag_plga_gemini.py` ！ Zotero auto-tagging via Gemini; still part of the maintained Stage0 toolchain in `project/PIPELINE_SCRIPT_MAP.md`.
- `src/stage0_relevance/auto_tag_plga_openai.py` ！ Zotero auto-tagging via OpenAI; same maintained Stage0 role.
- `src/stage0_relevance/classify_gemini_grouped.py` ！ grouped relevance classifier; explicit Stage0 pipeline component in the script map.
- `src/stage0_relevance/fill_missing_snapshots.py` ！ snapshot-repair utility; continuing dataset-maintenance value.
- `src/stage0_relevance/prefilter_regex.py` ！ deterministic prefilter; core reusable front-door filter.
- `src/stage0_relevance/zotero_api_sync_selected.py` ！ raw-index sync helper; clear continuing utility for metadata refresh.
- `src/stage0_relevance/zotero_fetch_llm_relevant_pdfs.py` ！ fetches source files for relevant papers; maintained acquisition utility.
- `src/stage0_relevance/zotero_llm_relevant_interactive.py` ！ interactive review helper for relevance triage; clearly reusable.
- `src/stage0_relevance/zotero_tag_sync.py` ！ tag synchronization tool; maintained Stage0 utility.

## Stage 1 cleaning

- `src/stage1_cleaning/clean_manifest_to_text.py` ！ manifest-to-cleaned-text transformer; named `ACTIVE_SUPPORTING` in the runbook.
- `src/stage1_cleaning/extract_tables_for_keys_v1.py` ！ table extraction worker invoked by the dataset runner; clear reusable worker contract.
- `src/stage1_cleaning/pdf2clean.py` ！ maintained PDF cleaner; explicit Stage1 component in the script map.
- `src/stage1_cleaning/run_tables_extraction_for_dataset_v1.py` ！ dataset-level table extraction runner; named `ACTIVE_SUPPORTING` in the runbook.
- `src/stage1_cleaning/zotero_raw_to_manifest.py` ！ raw-to-manifest builder; named `ACTIVE_SUPPORTING` in the runbook.

## Stage 2 support

- `src/stage2_sampling_labels/build_evidence_bundle_for_keys_v1.py` ！ deterministic evidence bundle builder; distinct reusable bridge between cleaned assets and audit workflows.
- `src/stage2_sampling_labels/build_key2txt_from_sample_manifest.py` ！ sample-local text index builder; named `ACTIVE_SUPPORTING` in the runbook.
- `src/stage2_sampling_labels/export_blockpack_audit_v7pilot_r3_fixparse.py` ！ current packer audit export; named `ACTIVE_SUPPORTING` in the runbook and tied to the active extractor.
- `src/stage2_sampling_labels/export_evidence_bundle_audit_xlsx_v1.py` ！ structured review export for evidence bundles; clear reusable output contract.
- `src/stage2_sampling_labels/sample_from_manifest_html_first.py` ！ maintained split/sample builder; named `ACTIVE_SUPPORTING` in the runbook.

## Stage 3 GT maintenance

- `src/stage3_gt/build_gt_template_from_conflict_queue.py` ！ manual GT template builder; explicitly supporting in `project/PIPELINE_SCRIPT_MAP.md`.
- `src/stage3_gt/export_gt_annotation_view.py` ！ annotation-view export helper; explicit supporting GT utility in the script map.
- `src/stage3_gt/gt_summary_report.py` ！ GT review summary/report utility; explicit supporting GT utility.
- `src/stage3_gt/merge_gt_from_annotation_view.py` ！ merge-back utility for reviewed annotations; explicit supporting GT utility.

## Stage 4 reusable deterministic evaluators

- `src/stage4_eval/compute_formulation_alignment_v1.py` ！ deterministic formulation alignment calculator; explicitly marked active in `project/PIPELINE_SCRIPT_MAP.md` and still conceptually central to formulation-level evaluation.
- `src/stage4_eval/compute_set_level_ee_match_v1.py` ！ generic set-level comparison utility with reusable I/O; narrower than mainline but still recognizable as a maintained evaluator.

## Stage 5 schema/export pipeline tools

- `src/stage5_benchmark/build_two_table_schema_v2.py` ！ maintained schema builder for the two-table representation; near-future Stage5 utility.
- `src/stage5_benchmark/build_two_table_schema_v3.py` ！ maintained DOE-aware schema builder; still part of active schema evolution rather than frozen history.
- `src/stage5_benchmark/derive_doe_coded_factors_v1.py` ！ deterministic DOE decoding utility; clear continuing value.
- `src/stage5_benchmark/export_full_database_v1.py` ！ full database exporter to `data/db/<db_version>/`; durable output contract.
- `src/stage5_benchmark/formulation_core_signature_v1.py` ！ Stage5 formulation-core signature logic; named `ACTIVE_SUPPORTING` in the runbook.
- `src/stage5_benchmark/run_alignment_eval_core_v1.py` ！ core-level alignment runner; reusable Stage5 evaluation utility.
- `src/stage5_benchmark/run_alignment_eval_schema_v3_v1.py` ！ schema-v3 alignment/evaluation runner; still useful for maintained schema comparisons.
- `src/stage5_benchmark/run_alignment_eval_v1.py` ！ generic alignment evaluation runner; clear reusable contract.
- `src/stage5_benchmark/run_core_eval_pipeline_v1.py` ！ orchestration wrapper for core evaluation; continuing utility while Stage5 remains active.
- `src/stage5_benchmark/run_derivation_v1.py` ！ derivation layer runner; durable Stage5 building block.
- `src/stage5_benchmark/run_evidence_realign_v1.py` ！ deterministic evidence realignment utility; current hardening-style utility, not just a one-off debug note.
- `src/stage5_benchmark/run_evidence_token_qc_v1.py` ！ evidence-token QC runner; durable field-level QC tool.
- `src/stage5_benchmark/run_formulation_core_signature_v1.py` ！ CLI runner for the core-signature module; clear maintained wrapper.
- `src/stage5_benchmark/run_projection_core_to_curated_v1.py` ！ projection runner from core schema to curated schema; clear reusable contract.
- `src/stage5_benchmark/run_projection_to_curated_v1.py` ！ projection runner from benchmark outputs to curated schema; reusable Stage5 utility.

## Stage 5 merge/publish

- `src/stage5_merge_publish/merge_results.py` ！ final result merger; explicit Stage5 pipeline component in `project/PIPELINE_SCRIPT_MAP.md`.

## Shared utilities

- `src/utils/build_dataset_split_dev_v1.py` ！ registered DEV split builder; continuing utility with current split-registry policy.
- `src/utils/build_dataset_split_test_v1.py` ！ TEST split builder with DEV exclusion; clear continuing utility.
- `src/utils/build_global_zotero_index_v1.py` ！ shared raw-index builder with clear repository-wide utility.
- `src/utils/gemini_models.py` ！ maintained model-name helper module.
- `src/utils/html_parser.py` ！ shared parser module reused across cleaning code.
- `src/utils/model_policy.py` ！ shared model-policy enforcement imported by multiple active scripts.
- `src/utils/paths.py` ！ canonical path resolver; named `ACTIVE_SUPPORTING` in the runbook.
- `src/utils/run_id.py` ！ run-id generation/validation helper; continuing run-discipline utility.
- `src/utils/run_latest.py` ！ latest-run pointer/fingerprint helper imported across active and Stage5 scripts.
- `src/utils/run_preflight.py` ！ run-preflight utility; named `ACTIVE_SUPPORTING` in the runbook.
- `src/utils/split_registry_v1.py` ！ active split-registry loader used by current split/layout tools.
- `src/utils/validate_dataset_layout_v1.py` ！ layout validator tied to the current dataset/split conventions.

# 4. Downgraded to ARCHIVED_METHOD

These files were previously treated as `STABLE_TOOL` in the first pass but do not clear the stricter stability bar.

- `src/stage2_sampling_labels/build_strata_tags_from_text.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is a narrow sampling-prep helper with weak governance evidence and no active-flow role; historical/comparative role: auxiliary sampling experiment support.
- `src/stage2_sampling_labels/sample10_from_zotero_manifest.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is tied to a historical sample family (`sample10`) rather than the current split discipline; historical/comparative role: early subset-construction baseline.
- `src/stage3_gt/build_dev15_formulation_skeleton_review_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is DEV15-specific and benchmark-workbook-specific rather than a general GT utility; historical/comparative role: curated DEV15 skeleton review workflow.
- `src/stage3_gt/export_dev15_formulation_skeleton_gt_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it exports one fixed review workbook family; historical/comparative role: DEV15 skeleton GT consolidation path.
- `src/stage3_gt/formulation_skeleton_common.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it only supports the DEV15 skeleton family and has no independent maintained contract; historical/comparative role: shared helper module for the archived DEV15 skeleton tooling.
- `src/stage3_gt/validate_dev15_formulation_skeleton_review_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it validates one specific DEV15 review workbook format; historical/comparative role: DEV15 skeleton quality gate.
- `src/stage4_eval/apply_extracted_ee_dedup_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it belongs to the February benchmark-optimization batch rather than the current mainline; historical/comparative role: extracted-row dedup experiment.
- `src/stage4_eval/apply_formulation_grouping_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because the runbook explicitly says not to default to it, and architecture marks grouping logic as an upstream-redesign candidate rather than stable deterministic core; historical/comparative role: historical grouping layer.
- `src/stage4_eval/apply_global_baseline_inheritance_and_rerun_alignment_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because the runbook explicitly treats it as non-default and the architecture treats this rule family as tolerated rather than stable core; historical/comparative role: inheritance/baseline reconstruction method.
- `src/stage4_eval/build_boundary_alignment_diagnostics_pack_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because its own CLI description is for a frozen DEV-15 diagnostic pack, not a general maintained utility; historical/comparative role: frozen benchmark boundary-audit pack.
- `src/stage4_eval/build_failure_profile_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is a narrow February benchmark-diagnostics artifact generator; historical/comparative role: failure-profiling experiment.
- `src/stage4_eval/build_goren_overlap_scaffold_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it scaffolds one benchmark overlap family rather than a general pipeline contract; historical/comparative role: Goren overlap setup method.
- `src/stage4_eval/build_per_doi_diagnostics_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is a narrow benchmark-diagnostics builder with weak present-day governance support; historical/comparative role: per-DOI diagnostic pack.
- `src/stage4_eval/compare_drugname_sets_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is a narrow comparison helper rather than a maintained stage utility; historical/comparative role: diagnostic normalization comparison.
- `src/stage4_eval/compute_alignment_sensitivity_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is sensitivity analysis for a specific alignment family, not a broadly maintained operational tool; historical/comparative role: parameter-comparison evaluator.
- `src/stage4_eval/compute_goren_metrics_tables_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is benchmark-report-table generation tied to a specific evaluation family; historical/comparative role: Goren metrics reporting helper.
- `src/stage4_eval/export_dev15_formulation_view_xlsx_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is explicitly DEV15-specific and superseded in practice by the current reviewer workbook path; historical/comparative role: older human-review export.
- `src/stage4_eval/precision_recovery_experiment_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because the file is explicitly an experiment and not a maintained workflow primitive; historical/comparative role: precision-recovery benchmark experiment.
- `src/stage4_eval/run_alignment_v3_surfactant_drugnorm.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is an older alignment variant family the runbook tells agents not to default to for current work; historical/comparative role: normalization/alignment baseline.
- `src/stage5_benchmark/analyze_row_membership_core_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is a narrow evaluation-report layer around one core-eval family rather than a repository-wide maintained utility; historical/comparative role: core multiplicity analysis.
- `src/stage5_benchmark/analyze_row_membership_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is a report-style benchmark analyzer rather than a core durable building block; historical/comparative role: projected-vs-curated row-membership analysis.
- `src/stage5_benchmark/build_goren_overlap_manifest.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is benchmark-specific setup logic for one overlap family; historical/comparative role: Goren overlap preparation.
- `src/stage5_benchmark/build_two_table_schema_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it has been overtaken by `schema_v2` and `schema_v3`; historical/comparative role: first released two-table schema baseline.
- `src/stage5_benchmark/copy_goren_dataset.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is a one-benchmark dataset copier rather than a general maintained tool; historical/comparative role: benchmark dataset bootstrap helper.
- `src/stage5_benchmark/evaluate_against_goren.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is benchmark-family-specific and weaker than the newer projection/alignment pipeline runners; historical/comparative role: earlier Goren evaluation wrapper.
- `src/stage5_benchmark/export_audit_to_excel_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is a convenience review exporter around one audit family, not a core maintained primitive; historical/comparative role: benchmark audit workbook export.
- `src/stage5_benchmark/inspect_schemas.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is an inspection helper for a specific weak-label TSV/JSONL pair, not a maintained stage utility; historical/comparative role: schema inspection probe.
- `src/stage5_benchmark/make_audit_pack_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it packages one audit family and is closer to branch tooling than to enduring infrastructure; historical/comparative role: parsing/derivation audit pack builder.
- `src/stage5_benchmark/report_schema_v2_v3_core_diff_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it exists to compare two historical schema variants rather than serve as a continuing pipeline primitive; historical/comparative role: schema-diff verification report.
- `src/utils/build_dataset_root_from_legacy_v1.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is explicitly about legacy-root migration; historical/comparative role: layout migration helper.
- `src/utils/build_ee_only_manifest.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is a narrow manifest transformation for one EE-focused slice, not a maintained default utility; historical/comparative role: EE-focused dataset prep helper.
- `src/utils/convert_sample_manifest.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is an ad hoc schema adapter for one sample-manifest family and overlaps with the TSV converter; historical/comparative role: sample-manifest conversion helper.
- `src/utils/convert_sample_manifest_to_tsv.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is another narrow adapter around one legacy key2txt contract; historical/comparative role: sample-manifest TSV conversion helper.
- `src/utils/scan_ee_coverage.py` ！ previous classification: `STABLE_TOOL`; not a true stable tool because it is a narrow pattern scanner for a single field family rather than a maintained pipeline primitive; historical/comparative role: EE-coverage exploratory scanner.

# 5. Downgraded to DELETE_CANDIDATE

These files were previously treated as `STABLE_TOOL`, but repository evidence is too weak to justify keeping them as maintained utilities.

- `src/stage4_eval/audit_top3_doi_root_cause_v1.py` ！ previous classification: `STABLE_TOOL`; evidence: filename encodes a one-off `top3` analysis target, the script uses frozen benchmark defaults, it is not named in the active flow/runbook/script map, and `docs/tool_index.md` itself also flags it as a likely iterative experiment; risk level: medium; recommendation: needs content check.
- `src/stage4_eval/inspect_formulation_view_inventory_v1.py` ！ previous classification: `STABLE_TOOL`; evidence: `inspect`/inventory debug role, no active governance references, no clear import/use relationships, and repo triage history already marked it as unregistered debug; risk level: medium; recommendation: needs content check.
- `src/stage5_benchmark/debug_doi_overlap.py` ！ previous classification: `STABLE_TOOL`; evidence: explicit `debug` identity, frozen Goren overlap defaults, no active governance references, no referencers in `src/`, and likely superseded by maintained overlap/evaluation tooling; risk level: medium; recommendation: needs content check.
- `src/utils/test_gemini.py` ！ previous classification: `STABLE_TOOL`; evidence: ad hoc connectivity probe script, hardcoded one-call behavior, no repository referencers, and no role in the governed pipeline; risk level: low; recommendation: safe after review.

# 6. Explicit review of dual-model / comparison / GT helper families

## Dual-model extraction/comparison family

Keep as archived:

- `src/stage4_eval/auto_extract_multimodel.py`
- `src/stage4_eval/multi_model_extract_tier1.py`
- `src/stage4_eval/multi_model_extract_tier2.py`
- `src/stage4_eval/multi_model_merge_qc.py`
- `src/stage4_eval/multi_model_consensus_vote.py`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v5_G3.py`

Reason:
These are historically meaningful comparative methods, but `project/ACTIVE_PIPELINE_RUNBOOK.md` explicitly says not to default to multimodel or older extraction families for current DEV work.

## Comparison and reconciliation family

Keep as stable tool:

- `src/stage4_eval/compute_formulation_alignment_v1.py`
- `src/stage4_eval/compute_set_level_ee_match_v1.py`

Downgrade to archived:

- `src/stage4_eval/compare_drugname_sets_v1.py`
- `src/stage4_eval/compute_alignment_sensitivity_v1.py`
- `src/stage4_eval/compare_v7pilot_scope_r1_r2.py`
- `src/stage4_eval/run_alignment_v3_surfactant_drugnorm.py`
- `src/stage4_eval/test_doe_coordinate_reconciliation_v1.py`
- `src/stage5_benchmark/report_schema_v2_v3_core_diff_v1.py`

Nominate for deletion review:

- `src/stage4_eval/audit_top3_doi_root_cause_v1.py`
- `src/stage5_benchmark/debug_doi_overlap.py`

Reason:
Only the first two still look like maintained generic evaluators. The rest are variant comparison, experimental reconciliation, or one-off debug/report paths.

## Manual GT helper family

Keep as active:

- `src/stage4_eval/build_dev15_review_workbook_v1.py`

Keep as stable tool:

- `src/stage3_gt/build_gt_template_from_conflict_queue.py`
- `src/stage3_gt/export_gt_annotation_view.py`
- `src/stage3_gt/merge_gt_from_annotation_view.py`
- `src/stage3_gt/gt_summary_report.py`

Downgrade to archived:

- `src/stage3_gt/build_dev15_formulation_skeleton_review_v1.py`
- `src/stage3_gt/validate_dev15_formulation_skeleton_review_v1.py`
- `src/stage3_gt/export_dev15_formulation_skeleton_gt_v1.py`
- `src/stage3_gt/formulation_skeleton_common.py`
- `src/legacy/20260202/gt_tool.py`
- `src/legacy/20260202/gt_tool_v3.py`
- `src/stage4_eval/export_dev15_formulation_view_xlsx_v1.py`
- `src/stage4_eval/export_dev15_dashboard_and_audit_pack_v1.py`

Reason:
The generic conflict-template / annotation / merge / summary loop still looks maintained. The DEV15 skeleton family is too benchmark-specific to count as a stable repository-wide tool.

## Audit-pack and one-off experiment family

Keep as stable tool:

- `src/stage2_sampling_labels/export_blockpack_audit_v7pilot_r3_fixparse.py`
- `src/stage2_sampling_labels/export_evidence_bundle_audit_xlsx_v1.py`
- `src/stage5_benchmark/run_evidence_token_qc_v1.py`
- `src/stage5_benchmark/run_evidence_realign_v1.py`

Downgrade to archived:

- `src/stage4_eval/build_boundary_alignment_diagnostics_pack_v1.py`
- `src/stage4_eval/build_failure_profile_v1.py`
- `src/stage4_eval/build_per_doi_diagnostics_v1.py`
- `src/stage4_eval/precision_recovery_experiment_v1.py`
- `src/stage5_benchmark/build_audit_pack_human_evidence_v1.py`
- `src/stage5_benchmark/build_doe_signature_injection_audit_pack_v1.py`
- `src/stage5_benchmark/build_signature_v1_audit_pack.py`
- `src/stage5_benchmark/export_human_optimization_audit_10_v1.py`
- `src/stage5_benchmark/run_audit10_regression_check_v1.py`

Nominate for deletion review:

- `src/stage4_eval/inspect_formulation_view_inventory_v1.py`

Reason:
Only the audit tools directly tied to the active extractor or durable evidence-QC rules survive as stable. The rest are branch-specific packs or inspection residue.

# 7. Proposed reduction-ready subsets

## Stable tools worth keeping in main stage directories

- Stage0 relevance acquisition and Zotero sync scripts.
- Stage1 manifest/text/table builders.
- Stage2 split/index/evidence support scripts directly adjacent to the active extractor.
- Generic GT maintenance helpers (`build_gt_template_from_conflict_queue.py`, `export_gt_annotation_view.py`, `merge_gt_from_annotation_view.py`, `gt_summary_report.py`).
- A reduced Stage4 reusable core centered on `compute_formulation_alignment_v1.py` and `compute_set_level_ee_match_v1.py`.
- The maintained Stage5 schema/projection/alignment/export runners.
- Shared utilities for paths, run discipline, split registry, model policy, and dataset split/layout validation.

## Archived methods worth later relocating to an archive subtree

- All older weak-label extractor variants other than `auto_extract_weak_labels_v7pilot_r3_fixparse.py`.
- All multimodel extraction/comparison scripts.
- DEV15 skeleton GT tooling.
- February benchmark diagnostics and comparison helpers.
- Narrow Goren overlap setup/evaluation helpers that are no longer the preferred Stage5 path.
- Legacy-root/sample-manifest conversion helpers and field-specific exploratory scanners.

## Strongest delete candidates for a future human-approved cleanup pass

- `src/utils/test_gemini.py`
- `src/stage4_eval/audit_top3_doi_root_cause_v1.py`
- `src/stage4_eval/inspect_formulation_view_inventory_v1.py`
- `src/stage5_benchmark/debug_doi_overlap.py`

# 8. Revised counts

- `ACTIVE_ENTRYPOINT`: 3
- `STABLE_TOOL`: 52
- `ARCHIVED_METHOD`: 73
- `DELETE_CANDIDATE`: 4
