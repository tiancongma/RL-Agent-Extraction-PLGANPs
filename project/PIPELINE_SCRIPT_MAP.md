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
| Stage 2 | Composite semantic extraction and deterministic post-LLM completion | `src/stage2_sampling_labels/run_stage2_composite_v1.py` | `ACTIVE_ENTRYPOINT` | Run the governed composite Stage2 graph: LLM semantic discovery from cleaned assets followed by deterministic post-LLM completion into the only authoritative Stage2 artifact consumed by Stage3, then refresh run-level feature-unit observability inside `RUN_CONTEXT.md`. Stage2 live input assembly can optionally use the governed ordered-evidence-pack mode when `STAGE2_INPUT_EVIDENCE_PACKING_MODE=ordered_blocks`. | scope manifest TSV; cleaned text; cleaned or governed tables; paper-key subset; source_mode; llm_backend; model; max_text_chars | semantic-intermediate JSONL/summary under `semantic_stage2_objects/`; completed Stage2 weak-label TSV/JSONL under `semantic_to_widerow_adapter/`; `RUN_CONTEXT.md` |
| Stage 3 | Deterministic formulation relation materialization | `src/stage3_relation/build_formulation_relation_artifacts_v1.py` | `ACTIVE_ENTRYPOINT` | Build explicit paper-level formulation relation artifacts and resolved relation-backed descriptive synthesis fields from the compatibility-projected legacy wide-row surface without any LLM usage. | compatibility-projected Stage 2 candidate formulation-instance TSV; optional compatibility-projected JSONL; optional scope manifest TSV | `formulation_relation_records_v1.tsv`; `formulation_logic_graph_v1.jsonl`; `formulation_relation_summary_v1.tsv`; `resolved_relation_fields_v1.tsv` |
| Stage 4 | Candidate-level diagnostics and review | `src/stage4_eval/eval_weak_labels_v7pilot3.py` | `ACTIVE_ENTRYPOINT` | Produce candidate-instance diagnostic counts and mismatch artifacts from the compatibility-projected legacy wide-row surface. | compatibility-projected Stage 2 candidate TSV; scope manifest; GT workbook | per-paper diagnostic TSVs and summary markdown |
| Stage 4 | Candidate-level diagnostics and review | `src/stage4_eval/build_dev15_review_workbook_v1.py` | `STABLE_TOOL` | Build reviewer-facing workbooks from Stage 4 artifacts. | Stage 4 summaries; checked manual workbook | reviewer workbook XLSX |
| Stage 5 | Final formulation closure and benchmark comparison | `src/stage5_benchmark/build_minimal_final_output_v1.py` | `ACTIVE_ENTRYPOINT` | Build the final formulation table and decision trace from compatibility-projected Stage 2 candidate rows by materializing direct extraction fields plus explicit Stage 3 resolved relation fields, while applying conservative benchmark-facing identity guardrails such as descendant filtering. Parent-linked non-synthesis descendants remain filterable, sweep-style `variant_formulation` members are not auto-filtered by `post_processing` alone, and parent-linked helper descendants with preserved blank/control/model-drug-substitution semantics are also filterable even when upstream routing tags regress. | compatibility-projected Stage 2 candidate formulation-instance TSV; required Stage 3 relation-record TSV; required Stage 3 resolved-relation-field TSV | `final_formulation_table_v1.tsv`; `final_output_decision_trace_v1.tsv`; `final_output_summary_v1.md` |
| Stage 5 | Comparison node | `src/stage5_benchmark/compare_final_table_to_gt_v1.py` | `ACTIVE_ENTRYPOINT` | Compare only the Stage 5 final formulation table to the checked GT workbook and refresh run-level feature-unit observability inside `RUN_CONTEXT.md`. | final formulation table; scope manifest; fixed GT workbook | `final_table_vs_gt_counts.tsv`; `final_table_vs_gt_summary.md`; `RUN_CONTEXT.md` |

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
- Stage2.5 is retired from the active mainline and remains archived only as
  historical exploratory design input.
- Corrective architecture freeze (`2026-03-30`):
  - LLM owns open semantic discovery and formulation-boundary discovery in
    Stage2.
  - Deterministic post-LLM completion remains inside Stage2.
  - Deterministic logic in Stage3+ owns relation resolution, inheritance
    handling, normalization, filtering, audit, and materialization.
  - Deterministic semantic emitters and semantic lifts may exist only as
    fallback, comparator, migration-support, or diagnostic infrastructure.
  - The legacy wide-row extractor is deprecated and must not replace the
    frozen LLM-centered Stage2 authority contract.

## Optional Diagnostic / Review Path

- `src/stage4_eval/eval_weak_labels_v7pilot3.py`
- `src/stage4_eval/build_dev15_review_workbook_v1.py`
- `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py`
- `src/stage5_benchmark/build_field_gt_review_workbook_v1.py`

These scripts inspect candidate-instance behavior and support review. They do
not define the production endpoint.

Current interpretation:

- the benchmark-valid production endpoint remains the Stage 5 final
  formulation table
- reviewer-facing Layer 2 and Layer 3 audit surfaces are still downstream of
  that endpoint
- however, these review surfaces are also part of the governed production
  audit and governance layer for the formulation database
- current repo capability is partially present but not yet unified into one
  formulation-centered audit system contract

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
| `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py` | `STABLE_TOOL` | Internal LLM semantic-discovery substep used by the governed composite Stage2 entrypoint. Emits object-first semantic-intermediate artifacts for any declared manifest scope. |
| `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py` | `STABLE_TOOL` | Internal deterministic post-LLM completion substep used by the governed composite Stage2 entrypoint. Converts semantic intermediates into the completed Stage2 artifact required by unchanged downstream consumers, including governed DOE row expansion and governed non-DOE table row expansion when authorized by Stage2 semantic markers. |
| `src/stage2_sampling_labels/table_row_expansion_v1.py` | `STABLE_TOOL` | Deterministically enumerate explicit non-DOE formulation-table rows from Stage1 table assets when the LLM has declared a formulation-bearing non-DOE table through the Stage2 table authorization contract. |
| `src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py` | `STABLE_TOOL` | Maintained Stage2 contract validator that checks semantic-source mode declaration, candidate provenance, and DOE expansion legality before a composite Stage2 output can stand as governed authority. |
| `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py` | `STABLE_TOOL` | Deterministic semantic Stage2 comparator or fallback retained for migration-support, comparator, or diagnostic work only; not the governed Stage2 authority. |
| `src/stage2_sampling_labels/extract_semantic_stage2_v2_threepaper.py` | `STABLE_TOOL` | Compatibility shim for the historical three-paper extractor name. Use the governed generic extractor instead; do not treat this filename as the Stage2 definition. |
| `src/stage2_sampling_labels/build_stage2_replacement_contract_v1.py` | `STABLE_TOOL` | Write the non-default Stage2 replacement semantic-contract scaffold artifacts used for redesign planning and compatibility mapping. It is not a benchmark runtime entrypoint. |
| `src/stage2_sampling_labels/enrich_preparation_method_fields_v1.py` | `STABLE_TOOL` | Deterministically append schema-only preparation-method enrichment fields to an existing Stage2-style TSV without changing row identity or counts. |
| `src/stage2_sampling_labels/export_blockpack_audit_v7pilot_r3_fixparse.py` | `STABLE_TOOL` | Export the Stage 2 evidence packing order for manual audit. |
| `src/stage2_sampling_labels/export_evidence_bundle_audit_xlsx_v1.py` | `STABLE_TOOL` | Export evidence-bundle audit views. |
| `src/stage2_sampling_labels/run_targeted_stage2_regression_v1.py` | `STABLE_TOOL` | Controlled Stage 2 regression runner for diagnostic-only work. |
| `src/stage2_sampling_labels/diagnose_5gif3d8w_root_cause_v1.py` | `STABLE_TOOL` | Paper-specific diagnostic helper retained for regression analysis. |
| `src/stage2_sampling_labels/diagnose_5gif3d8w_axis_applicability_v1.py` | `STABLE_TOOL` | Axis-applicability diagnostic helper retained for regression analysis. |

Legacy extractor note:

- The deprecated wide-row fallback extractor carried historical evidence-packing
  and instance-kind reconciliation behavior.
- Those details remain relevant only for fallback or historical comparison work.
- They are not the authoritative Stage2 contract for the active semantic
  Stage2 -> adapter -> Stage3 -> Stage5 mainline.

Stage 2.5 archival note:

- `src/stage2_5_components_shadow/build_text_evidence_packs_v0.py` is
  retained as an archived exploratory Stage2.5A shadow evidence-pack builder.
- It is non-authoritative and must not feed Stage3 or Stage5.

Stage 2 active contract note:

- The frozen Stage2 authority contract is LLM semantic discovery, not
  deterministic semantic reconstruction.
- `src/stage2_sampling_labels/run_stage2_composite_v1.py` is the one governed
  Stage2 execution entrypoint.
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py` is the
  internal LLM semantic-discovery substep.
- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py` is
  the internal deterministic post-LLM completion substep.
- `src/stage2_sampling_labels/table_row_expansion_v1.py` is a governed
  deterministic Stage2 helper that expands non-DOE formulation tables only
  after the LLM has declared table-level authorization markers.
- `src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py`
  is the maintained contract guard that fails if semantic-source mode is
  undeclared, if deterministic expansion lacks authorized scope, or if DOE rows
  appear without LLM-declared DOE scope in llm-first mode. The same guard also
  requires non-DOE table-expanded rows to reference an LLM-declared table
  formulation authorization scope.
- `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`
  is retained only for fallback, comparator, migration-support, or diagnostic
  work and must not be treated as Stage2 authority.
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
  is legacy and deprecated.

### Archived Stage 2.5

| Script path | Class | Purpose |
|---|---|---|
| `src/stage2_5_components_shadow/build_text_evidence_packs_v0.py` | `STABLE_TOOL` | Archived exploratory Stage2.5A shadow evidence-pack builder retained for design reference only; not part of the active benchmark pipeline. |
| `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py` | `ARCHIVED_METHOD` | Deprecated legacy wide-row extractor retained only for fallback or debug scenarios outside the active mainline. |

### Stage 3

| Script path | Class | Purpose |
|---|---|---|
| `src/stage3_relation/run_formulation_relation_artifacts_v1.py` | `STABLE_TOOL` | Run the Stage 3 relation builder in a reproducible run-scoped results directory with explicit `RUN_CONTEXT.md`, including `resolved_relation_fields_v1.tsv`, then refresh feature-unit observability metadata for the run. |

### Stage 4

| Script path | Class | Purpose |
|---|---|---|
| `src/stage4_eval/compute_formulation_alignment_v1.py` | `STABLE_TOOL` | Deterministic formulation alignment checks. |
| `src/stage4_eval/inspect_formulation_view_inventory_v1.py` | `STABLE_TOOL` | Inspect reviewer-facing formulation view coverage. |
| `src/stage4_eval/apply_extracted_ee_dedup_v1.py` | `STABLE_TOOL` | Targeted EE dedup utility for review or diagnostic work. |
| `src/stage4_eval/audit_top3_doi_root_cause_v1.py` | `STABLE_TOOL` | Focused root-cause audit helper. |
| `src/stage4_eval/compute_set_level_ee_match_v1.py` | `STABLE_TOOL` | Set-level EE comparison support. |
| `src/stage4_eval/precision_recovery_experiment_v1.py` | `STABLE_TOOL` | Precision-recovery experiment helper; not a canonical stage endpoint. |
| `src/analysis/build_stage2_v2_threepaper_comparison_pack.py` | `STABLE_TOOL` | Build a governed three-paper Stage2-only comparison pack contrasting the minimal LLM Stage2 v2 slice against current deterministic semantic and historical wide-row comparator surfaces. |

### Stage 5

| Script path | Class | Purpose |
|---|---|---|
| `src/stage5_benchmark/run_minimal_final_output_v1.py` | `STABLE_TOOL` | NON-CANONICAL, STAGE5_ONLY convenience wrapper for Stage 5A closure only. It requires Stage 3 relation records plus resolved relation fields, refreshes feature-unit observability metadata, and is not a production-path entrypoint or hidden full-pipeline orchestrator. |
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
| `src/stage5_benchmark/export_final_formulation_audit_ready_v1.py` | `STABLE_TOOL` | Postprocess the Stage 5 final table into a reviewer-facing formulation audit surface for the downstream audit/governance layer without changing benchmark counts. |
| `src/stage5_benchmark/audit_evidence_resolver_v1.py` | `STABLE_TOOL` | Resolve paper-local text/table evidence pointers for downstream audit-pack and field-review tooling. |
| `src/stage5_benchmark/build_audit_pack_human_evidence_v1.py` | `STABLE_TOOL` | Build a human-readable evidence workbook for review of extracted formulation fields and provenance. |
| `src/stage5_benchmark/export_full_database_v1.py` | `STABLE_TOOL` | Final database export utility for downstream release work. |
| `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py` | `STABLE_TOOL` | Build a run-scoped XLSX review workbook for Layer 2 boundary GT from the Stage 5 final formulation table, with prediction-reference columns separated from GT-authoritative reviewer fields. |
| `src/stage5_benchmark/build_field_gt_review_workbook_v1.py` | `STABLE_TOOL` | Build a run-scoped XLSX review workbook for formulation-row value credibility audit from frozen Stage 5 final rows, with compact reviewer columns, helper formulation labels, Layer 2 paper-risk metadata, dropdown GT controls, and strict evidence/value support gating. |
| `src/stage5_benchmark/build_value_gt_annotation_workbook_v1.py` | `STABLE_TOOL` | Build a run-scoped formulation-level value annotation workbook by pivoting the Layer 3 field-review seed into one row per frozen formulation for fast manual numeric credibility review. The latest Stage5 final table plus audit-ready export are canonical for current-system presence; historical scaffold and prior-workbook bridge artifacts are advisory mapping aids only. |
| `src/stage5_benchmark/build_layer2_identity_scaffold_binding_v1.py` | `STABLE_TOOL` | Build a diagnostic-only scaffold-binding surface from a frozen Layer2-style GT workbook and selected Stage5 final tables using article-native-first identity binding. It emits audit TSV/markdown surfaces only and does not mutate benchmark-valid outputs. |
| `src/stage5_benchmark/enforce_identity_freeze_v1.py` | `STABLE_TOOL` | Mandatory Stage5 post-materialization identity gate. Validate `IDENTITY_FREEZE_RULE_V1` against an upstream identity scaffold plus a Stage5 final table, emit violation diagnostics, and fail non-zero on any identity-invariance breach unless explicitly run in report-only mode. |
| `src/stage5_benchmark/run_layer3_cross_audit_v1.py` | `STABLE_TOOL` | Build a report-only Layer 3 cross-audit pack from the compact value workbook plus cleaned text/tables. It emits deterministic cell-risk flags, bounded Gemini/NVIDIA auditor execution or task exports, partial backend checkpoints, and a merged human-review TSV/markdown report for value-credibility audit without modifying workbook contents or benchmark-valid artifacts. |
| `src/stage5_benchmark/build_layer2_risk_assessment_v1.py` | `STABLE_TOOL` | Build run-scoped Layer 2 paper-risk labels from an existing Layer 2 identity-comparison TSV for downstream Layer 3 audit prioritization, without changing benchmark-valid final outputs. |
| `src/stage5_benchmark/validate_layer3_evidence_contract_v1.py` | `STABLE_TOOL` | Validate Layer 3 reviewer-surface evidence handoff behavior against golden regression cases without changing benchmark-valid outputs. |
| `src/stage5_benchmark/validate_stage5_descendant_filter_regression_v1.py` | `STABLE_TOOL` | Deterministically rerun Stage 5 on frozen artifact inputs to validate the descendant-filter safeguards: BB3JUVW7 sweep-style variants must be retained, BXCV5XWB helper descendants must be filtered, nearby helper-control regressions such as RHMJWZX8 stay suppressed, and stable controls such as WIVUCMYG remain unchanged. |

### Cross-cutting governance support

| Script path | Class | Purpose |
|---|---|---|
| `src/utils/audit_run_lineage_layout_v1.py` | `STABLE_TOOL` | Deterministically audit top-level `data/results/run_*` lineage sprawl and flag sibling runs that should likely be contained under one parent lineage. |
| `src/utils/build_feature_activation_report_v1.py` | `STABLE_TOOL` | Build a run-scoped feature activation report from deterministic artifact evidence so child-lineage validation can be distinguished from parent-run activation and Stage2 ordered-input execution can be audited. |
| `src/utils/build_feature_execution_ledger_v1.py` | `STABLE_TOOL` | Build a full per-run feature execution ledger from the recovered feature inventory plus governed run artifacts so expected-vs-actual activation, upstream-applied processing, replay-hidden visibility limits, and resume-boundary scope can be checked before semantic debugging. |
| `src/utils/run_threepaper_stage2_v2_comparison.py` | `STABLE_TOOL` | Non-default three-paper semantic-intermediate comparison wrapper. It calls the governed composite Stage2 entrypoint, then builds a comparison pack for `WIVUCMYG`, `UFXX9WXE`, and `5GIF3D8W` without defining Stage2 itself or modifying `ACTIVE_RUN.json`. |
| `src/utils/build_mem_v1.py` | `STABLE_TOOL` | Build the governed `data/mem/v1/` memory registry from `docs/snapshots/`, `docs/methods/`, `data/results/**/RUN_CONTEXT.md`, and `project/*.md` without creating a new pipeline stage. |
| `src/utils/query_mem_v1.py` | `STABLE_TOOL` | Query the governed `data/mem/v1/` registry by text, type, stage, or run before deeper debugging or failure-localization work. |
| `src/utils/mem_bootstrap_v1.py` | `STABLE_TOOL` | Bootstrap complex-task memory lookup by classifying the task, surfacing relevant `mem_v1` hits, and suggesting the next governed files to read. |
| `src/utils/update_mem_v1.py` | `STABLE_TOOL` | Append a targeted governed memory row to `data/mem/v1/` without silent overwrite of existing logical entries. |
| `src/utils/check_mem_v1.py` | `STABLE_TOOL` | Validate `data/mem/v1/` schema, row references, ID prefixes, and path-length constraints. |
| `src/utils/update_run_context_with_feature_activation_v1.py` | `STABLE_TOOL` | Refresh a run's `RUN_CONTEXT.md` with feature-unit activation metadata, activation-state status, and a deterministic activation gate. |

Supporting-memory rule:

- `data/mem/v1/` is a supporting memory layer only and must not be treated as Stage 0-5 pipeline output.
- for complex debugging, regression, run comparison, GT mismatch analysis, pipeline modification, or lineage tracing, query memory before deeper repository exploration

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
