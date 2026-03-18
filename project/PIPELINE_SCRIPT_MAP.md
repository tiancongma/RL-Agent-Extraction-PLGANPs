# Pipeline Script Map

This document maps the canonical active pipeline to Stage 0 through Stage 5
only.

It exists to answer four questions:

1. which scripts are part of the canonical manual path
2. which scripts are reusable stage-local tools
3. which scripts are historical only
4. which scripts are not allowed to stand in for hidden orchestration

The map distinguishes three things explicitly:

- the production path
- the optional diagnostic / review path
- the evaluation reference path
- the comparison node

Script classes used here are fixed:

- `ACTIVE_ENTRYPOINT`
- `STABLE_TOOL`
- `ARCHIVED_METHOD`

## Canonical Stage 0 To Stage 5 Entry Points

| Stage | Stage name | Script path | Class | Purpose | Primary inputs | Primary outputs |
|---|---|---|---|---|---|---|
| Stage 0 | Relevance filtering and raw corpus intake | `src/stage0_relevance/zotero_api_sync_selected.py` | `ACTIVE_ENTRYPOINT` | Sync the selected Zotero item set into the raw JSONL artifact used by downstream stages. | Zotero library selection; local storage root | `data/raw/zotero/zotero_selected_items.jsonl` |
| Stage 0 | Relevance filtering and raw corpus intake | `src/stage0_relevance/zotero_fetch_llm_relevant_pdfs.py` | `ACTIVE_ENTRYPOINT` | Fetch local PDF or HTML assets for the selected relevant records. | Zotero-tagged relevant items; DOI or attachment metadata | local source files referenced from the raw JSONL |
| Stage 1 | Manifest, clean text, and tables | `src/stage1_cleaning/zotero_raw_to_manifest.py` | `ACTIVE_ENTRYPOINT` | Convert the raw Zotero-derived JSONL into the authoritative manifest. | `data/raw/zotero/zotero_selected_items.jsonl` | `data/cleaned/index/manifest_current.tsv` |
| Stage 1 | Manifest, clean text, and tables | `src/stage1_cleaning/clean_manifest_to_text.py` | `ACTIVE_ENTRYPOINT` | Build cleaned text assets and the authoritative key-to-text mapping. | `data/cleaned/index/manifest_current.tsv` | `data/cleaned/content/text/`; `data/cleaned/index/key2txt.tsv` |
| Stage 1 | Manifest, clean text, and tables | `src/stage1_cleaning/run_tables_extraction_for_dataset_v1.py` | `ACTIVE_ENTRYPOINT` | Build dataset-local table assets for extraction and later audit. | dataset manifest TSV; cleaned content | dataset-local `tables/` assets |
| Stage 2 | Candidate formulation-instance extraction | `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py` | `ACTIVE_ENTRYPOINT` | Produce high-recall candidate formulation-instance rows from cleaned assets. | scope manifest TSV; cleaned text; optional tables; model access | run-scoped weak-label TSV and JSONL |
| Stage 3 | Deterministic formulation relation materialization | `src/stage3_relation/build_formulation_relation_artifacts_v1.py` | `ACTIVE_ENTRYPOINT` | Build explicit paper-level formulation relation artifacts and resolved relation-backed descriptive synthesis fields from Stage 2 weak labels without any LLM usage. | Stage 2 candidate formulation-instance TSV; optional Stage 2 JSONL; optional scope manifest TSV | `formulation_relation_records_v1.tsv`; `formulation_logic_graph_v1.jsonl`; `formulation_relation_summary_v1.tsv`; `resolved_relation_fields_v1.tsv` |
| Stage 4 | Candidate-level diagnostics and review | `src/stage4_eval/eval_weak_labels_v7pilot3.py` | `ACTIVE_ENTRYPOINT` | Produce candidate-instance diagnostic counts and mismatch artifacts. | Stage 2 candidate TSV; scope manifest; GT workbook | per-paper diagnostic TSVs and summary markdown |
| Stage 4 | Candidate-level diagnostics and review | `src/stage4_eval/build_dev15_review_workbook_v1.py` | `STABLE_TOOL` | Build reviewer-facing workbooks from Stage 4 artifacts. | Stage 4 summaries; checked manual workbook | reviewer workbook XLSX |
| Stage 5 | Final formulation closure and benchmark comparison | `src/stage5_benchmark/build_minimal_final_output_v1.py` | `ACTIVE_ENTRYPOINT` | Build the final formulation table and decision trace from Stage 2 candidate rows by materializing direct extraction fields plus explicit Stage 3 resolved relation fields, without Stage 5 semantic inference. | Stage 2 candidate formulation-instance TSV; required Stage 3 relation-record TSV; required Stage 3 resolved-relation-field TSV | `final_formulation_table_v1.tsv`; `final_output_decision_trace_v1.tsv`; `final_output_summary_v1.md` |
| Stage 5 | Comparison node | `src/stage5_benchmark/compare_final_table_to_gt_v1.py` | `ACTIVE_ENTRYPOINT` | Compare only the Stage 5 final formulation table to the checked GT workbook. | final formulation table; scope manifest; fixed GT workbook | `final_table_vs_gt_counts.tsv`; `final_table_vs_gt_summary.md` |

## Evaluation Reference Path

The manual GT assets are reference inputs, not internal production
transformations.

- primary current reference input:
  - `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v2_variantaware.xlsx`
- `src/stage3_relation/` is the active deterministic Stage 3 runtime namespace
- `src/stage3_gt/` remains a reserved reference namespace with no active routine runtime entrypoint
- historical GT-maintenance helpers are archived under `archive/code/`
- the comparison node reads both:
  - the production-path final formulation table
  - the fixed manual GT workbook

## Production Path Notes

- Canonical pipeline means explicit provenance from Zotero-selected inputs to
  `final_formulation_table_v1.tsv`.
- Stage 3 is part of the production path and now has a dedicated deterministic
  runtime entrypoint for explicit relation materialization.
- Stage 4 remains a diagnostic branch off the production path, not the
  production endpoint.
- The comparison node is downstream of the production path and uses the fixed
  GT workbook as a separate reference input.

## Optional Diagnostic / Review Path

- `src/stage4_eval/eval_weak_labels_v7pilot3.py`
- `src/stage4_eval/build_dev15_review_workbook_v1.py`
- `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py`
- `src/stage5_benchmark/build_field_gt_review_workbook_v1.py`

These scripts inspect candidate-instance behavior and support review. They do
not define the production endpoint.

## Stage-Local Stable Tools

These scripts are active engineering assets but are not themselves canonical
stage-completion entrypoints.

### Stage 0

| Script path | Class | Purpose |
|---|---|---|
| `src/stage0_relevance/prefilter_regex.py` | `STABLE_TOOL` | Cheap metadata prefilter before heavier relevance review. |
| `src/stage0_relevance/classify_gemini_grouped.py` | `STABLE_TOOL` | Grouped LLM relevance screening over metadata. |
| `src/stage0_relevance/auto_tag_plga_gemini.py` | `STABLE_TOOL` | Gemini-based Zotero tag helper. |
| `src/stage0_relevance/auto_tag_plga_openai.py` | `STABLE_TOOL` | OpenAI-based Zotero tag helper. |
| `src/stage0_relevance/zotero_tag_sync.py` | `STABLE_TOOL` | Sync local tagging state back to Zotero. |
| `src/stage0_relevance/zotero_llm_relevant_interactive.py` | `STABLE_TOOL` | Interactive relevance review helper. |
| `src/stage0_relevance/fill_missing_snapshots.py` | `STABLE_TOOL` | Fill missing HTML snapshot coverage. |

### Stage 1

| Script path | Class | Purpose |
|---|---|---|
| `src/stage1_cleaning/find_html_table_candidates_v1.py` | `STABLE_TOOL` | Probe HTML content for table candidates before extraction. |
| `src/stage1_cleaning/extract_tables_for_keys_v1.py` | `STABLE_TOOL` | Targeted table extraction for selected keys. |
| `src/stage1_cleaning/pdf2clean.py` | `STABLE_TOOL` | Underlying PDF or HTML cleaner used by Stage 1 wrappers. |
| `src/utils/html_parser.py` | `STABLE_TOOL` | Shared HTML parsing utilities. |

### Stage 2

| Script path | Class | Purpose |
|---|---|---|
| `src/stage2_sampling_labels/sample_from_manifest_html_first.py` | `STABLE_TOOL` | Build reproducible split or sample manifests. |
| `src/stage2_sampling_labels/build_key2txt_from_sample_manifest.py` | `STABLE_TOOL` | Build sample-local key-to-text mappings. |
| `src/stage2_sampling_labels/build_evidence_bundle_for_keys_v1.py` | `STABLE_TOOL` | Build deterministic evidence packages for selected keys. |
| `src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py` | `STABLE_TOOL` | Deterministically enumerate explicit numbered DOE formulation rows from Stage1 table assets and emit additive Stage2 candidate artifacts. |
| `src/stage2_sampling_labels/enrich_preparation_method_fields_v1.py` | `STABLE_TOOL` | Deterministically append schema-only preparation-method enrichment fields to an existing Stage2-style TSV without changing row identity or counts. |
| `src/stage2_sampling_labels/export_blockpack_audit_v7pilot_r3_fixparse.py` | `STABLE_TOOL` | Export the Stage 2 evidence packing order for manual audit. |
| `src/stage2_sampling_labels/export_evidence_bundle_audit_xlsx_v1.py` | `STABLE_TOOL` | Export evidence-bundle audit views. |
| `src/stage2_sampling_labels/run_targeted_stage2_regression_v1.py` | `STABLE_TOOL` | Controlled Stage 2 regression runner for diagnostic-only work. |
| `src/stage2_sampling_labels/diagnose_5gif3d8w_root_cause_v1.py` | `STABLE_TOOL` | Paper-specific diagnostic helper retained for regression analysis. |
| `src/stage2_sampling_labels/diagnose_5gif3d8w_axis_applicability_v1.py` | `STABLE_TOOL` | Axis-applicability diagnostic helper retained for regression analysis. |

### Stage 3

| Script path | Class | Purpose |
|---|---|---|
| `src/stage3_relation/run_formulation_relation_artifacts_v1.py` | `STABLE_TOOL` | Run the Stage 3 relation builder in a reproducible run-scoped results directory with explicit `RUN_CONTEXT.md`, including `resolved_relation_fields_v1.tsv`. |

### Stage 4

| Script path | Class | Purpose |
|---|---|---|
| `src/stage4_eval/compute_formulation_alignment_v1.py` | `STABLE_TOOL` | Deterministic formulation alignment checks. |
| `src/stage4_eval/inspect_formulation_view_inventory_v1.py` | `STABLE_TOOL` | Inspect reviewer-facing formulation view coverage. |
| `src/stage4_eval/apply_extracted_ee_dedup_v1.py` | `STABLE_TOOL` | Targeted EE dedup utility for review or diagnostic work. |
| `src/stage4_eval/audit_top3_doi_root_cause_v1.py` | `STABLE_TOOL` | Focused root-cause audit helper. |
| `src/stage4_eval/compute_set_level_ee_match_v1.py` | `STABLE_TOOL` | Set-level EE comparison support. |
| `src/stage4_eval/precision_recovery_experiment_v1.py` | `STABLE_TOOL` | Precision-recovery experiment helper; not a canonical stage endpoint. |

### Stage 5

| Script path | Class | Purpose |
|---|---|---|
| `src/stage5_benchmark/run_minimal_final_output_v1.py` | `STABLE_TOOL` | NON-CANONICAL, STAGE5_ONLY convenience wrapper for Stage 5A closure only. It requires Stage 3 relation records plus resolved relation fields and is not a production-path entrypoint or hidden full-pipeline orchestrator. |
| `src/stage5_benchmark/formulation_core_signature_v1.py` | `STABLE_TOOL` | Core-signature utility for downstream schema and database work. |
| `src/stage5_benchmark/build_two_table_schema_v2.py` | `STABLE_TOOL` | Schema builder for downstream database-facing table work. |
| `src/stage5_benchmark/build_two_table_schema_v3.py` | `STABLE_TOOL` | Newer schema builder for downstream database-facing table work. |
| `src/stage5_benchmark/run_formulation_core_signature_v1.py` | `STABLE_TOOL` | Runner for explicit core-signature generation. |
| `src/stage5_benchmark/run_derivation_v1.py` | `STABLE_TOOL` | Deterministic derivation helper for downstream tables. |
| `src/stage5_benchmark/run_projection_to_curated_v1.py` | `STABLE_TOOL` | Projection helper into curated table forms. |
| `src/stage5_benchmark/run_projection_core_to_curated_v1.py` | `STABLE_TOOL` | Projection helper from core signatures to curated exports. |
| `src/stage5_benchmark/run_alignment_eval_v1.py` | `STABLE_TOOL` | Alignment-evaluation helper for Stage 5 assets. |
| `src/stage5_benchmark/run_alignment_eval_core_v1.py` | `STABLE_TOOL` | Core-signature alignment evaluation helper. |
| `src/stage5_benchmark/run_alignment_eval_schema_v3_v1.py` | `STABLE_TOOL` | Schema-v3 alignment evaluation helper. |
| `src/stage5_benchmark/run_evidence_token_qc_v1.py` | `STABLE_TOOL` | Evidence-token QC helper for numeric field support and field-level review prioritization. |
| `src/stage5_benchmark/export_final_formulation_audit_ready_v1.py` | `STABLE_TOOL` | Postprocess the Stage 5 final table into a reviewer-facing audit surface without changing benchmark counts. |
| `src/stage5_benchmark/audit_evidence_resolver_v1.py` | `STABLE_TOOL` | Resolve paper-local text/table evidence pointers for downstream audit-pack and field-review tooling. |
| `src/stage5_benchmark/build_audit_pack_human_evidence_v1.py` | `STABLE_TOOL` | Build a human-readable evidence workbook for review of extracted formulation fields and provenance. |
| `src/stage5_benchmark/export_full_database_v1.py` | `STABLE_TOOL` | Final database export utility for downstream release work. |
| `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py` | `STABLE_TOOL` | Build a run-scoped XLSX review workbook for Layer 2 boundary GT from the Stage 5 final formulation table, with prediction-reference columns separated from GT-authoritative reviewer fields. |
| `src/stage5_benchmark/build_field_gt_review_workbook_v1.py` | `STABLE_TOOL` | Build a run-scoped XLSX review workbook for Layer 3 field GT from frozen Stage 5 final rows, with compact reviewer columns, helper formulation labels, dropdown GT controls, and strict evidence/value support gating. |

### Cross-cutting governance support

| Script path | Class | Purpose |
|---|---|---|
| `src/utils/audit_run_lineage_layout_v1.py` | `STABLE_TOOL` | Deterministically audit top-level `data/results/run_*` lineage sprawl and flag sibling runs that should likely be contained under one parent lineage. |
| `src/utils/build_feature_activation_report_v1.py` | `STABLE_TOOL` | Build a run-scoped feature activation report from deterministic artifact evidence so child-lineage validation can be distinguished from parent-run activation. |
| `src/utils/update_run_context_with_feature_activation_v1.py` | `STABLE_TOOL` | Refresh a run's `RUN_CONTEXT.md` with feature-unit activation metadata and a deterministic activation gate. |

## Archived Historical Methods

Historical methods live under `archive/code/` or
`archive/delete_candidates_pending_confirmation/`.

They are not part of the canonical runtime path and must not be revived by
default.

Representative archived families:

- `archive/code/dev15_skeleton_bootstrap/`
- `archive/code/stage4_rule_heavy_formulation_reconstruction/`
- `archive/code/older_weak_label_pilot_variants/`
- `archive/code/dual_model_extraction_comparison/`
- `archive/code/old_gt_arbitration/`
- `archive/code/stage5_merge_publish/`

All scripts in those locations are `ARCHIVED_METHOD` unless a future decision
explicitly promotes one back into an active stage namespace.

## Boundary Rules

- There is no active end-to-end orchestration Python script in the canonical
  path.
- Manual reproduction is defined only by `project/ACTIVE_PIPELINE_FLOW.md`.
- `src/` contains only active stage code and stable stage-local tools.
- Archived or delete-candidate code must remain outside `src/`.
- The only benchmark-valid comparison artifact is produced by Stage 5 from the
  final formulation table.
- Top-level `data/results/run_*` directories are reserved for independent
  lineages; same-lineage retries and repair steps belong under the parent
  lineage directory.
