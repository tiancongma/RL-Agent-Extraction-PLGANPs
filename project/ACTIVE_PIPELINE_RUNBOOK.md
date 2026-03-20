# Active Pipeline Runbook

This runbook explains how to execute the canonical pipeline manually, stage by
stage.

It does not define a hidden orchestration layer. The authoritative execution
order lives in `project/ACTIVE_PIPELINE_FLOW.md`, and this file explains how to
apply that flow in practice with correct run discipline, provenance, and
incremental reuse.

This runbook distinguishes:

- the production path
- the optional diagnostic / review path
- the evaluation reference path
- the final comparison node

## Read Order

1. `project/ACTIVE_PIPELINE_FLOW.md`
2. `project/PIPELINE_SCRIPT_MAP.md`
3. `project/4_DECISIONS_LOG.md`
4. `docs/tool_index.md`

## Recent Changes (2026-03-19)

- Canonical polymer MW field:
  - `polymer_mw_kDa` is now the canonical field name.
  - `plga_mw_kDa` is retained only as a legacy read alias for compatibility with older artifacts.
  - This is a naming correction only; the field meaning did not change.
- Stage 2 evidence packing:
  - Stage 2 now promotes a `materials_procurement` block type for shared/default procurement-style parameters.
  - Effective packing order is:
    - `metadata`
    - `synthesis_method`
    - `materials_procurement`
    - `table`
    - `caption`
    - `paragraph`
- Non-change reminder:
  - Stage 5 remains materialization-only.
  - The relation-first Stage 3 -> Stage 5 contract is unchanged.
- Validation note:
  - Stage 2 LLM input has changed; regression runs using fresh LLM calls are required to fully validate behavior.

## Operating Principle

A complete pipeline run means the declared scope has passed through every
required production stage from upstream corpus inputs to:

- `final_formulation_table_v1.tsv`

Benchmark-valid reporting then additionally requires the comparison node, which
reads:

- the final formulation table
- the fixed manual GT workbook
- the declared scope manifest

Complete pipeline does not mean forced full recomputation.

## Complete Pipeline vs Full Recomputation

These are not the same thing.

- Complete pipeline means every required canonical stage is covered by a valid,
  traceable artifact chain.
- Full recomputation means every upstream stage is re-run from scratch.

Allowed:

- reuse unchanged Stage 0 raw records
- reuse unchanged Stage 1 cleaned assets
- reuse unchanged Stage 2 candidate TSVs

Not allowed:

- skip a required stage entirely
- substitute undocumented files for canonical stage completion artifacts
- report a partial-layer artifact as if it were the final system result

## Manual Stage-By-Stage Execution

Follow the exact order in `project/ACTIVE_PIPELINE_FLOW.md`.

### Stage 0

Use Stage 0 only when raw Zotero-derived records or local full-text assets must
be built or refreshed.

Core scripts:

- `src/stage0_relevance/zotero_api_sync_selected.py`
- `src/stage0_relevance/zotero_fetch_llm_relevant_pdfs.py`

Completion artifact:

- `data/raw/zotero/zotero_selected_items.jsonl`

### Stage 1

Use Stage 1 when the authoritative manifest, cleaned text, or table assets are
missing or stale.

Core scripts:

- `src/stage1_cleaning/zotero_raw_to_manifest.py`
- `src/stage1_cleaning/clean_manifest_to_text.py`
- `src/stage1_cleaning/run_tables_extraction_for_dataset_v1.py`

Completion artifacts:

- `data/cleaned/index/manifest_current.tsv`
- `data/cleaned/index/key2txt.tsv`
- dataset-local cleaned text and table assets

### Stage 2

Use Stage 2 to produce the candidate formulation-instance TSV for the declared
scope.

Core script:

- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`

Supporting deterministic Stage2-boundary tool:

- `src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py`

Completion artifact:

- `data/results/<stage2_run_id>/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`

Additive Stage2 augmentation artifacts when numbered DOE tables are detected:

- `data/results/<stage2_run_id>/weak_labels_v7pilot_r3_fixparse/numbered_doe_row_candidates_v1.tsv`
- `data/results/<stage2_run_id>/weak_labels_v7pilot_r3_fixparse/numbered_doe_row_candidates_summary_v1.tsv`

Stage2 boundary rule:

- explicit numbered DOE or design-table rows belong to the upstream extraction boundary
- when such rows are present in Stage1 table assets, the deterministic enumerator may add missing candidates before Stage3 and Stage5
- downstream stages must not be expected to reconstruct those rows if Stage2 omitted them

### Stage 3

Stage 3 is the deterministic formulation relation-materialization boundary.

Runtime rule:

- Stage 3 belongs to the production path
- it converts Stage 2 candidate formulation-instance rows into explicit
  paper-level relation artifacts suitable for audit and downstream closure
- it must not call any LLM or external API
- it exists to separate relation reasoning from final flattening

Current production-boundary input:

- `data/results/<stage2_run_id>/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`
- optional Stage 2 JSONL and scope manifest TSV

Current production-boundary output:

- `formulation_relation_records_v1.tsv`
- `formulation_logic_graph_v1.jsonl`
- `formulation_relation_summary_v1.tsv`
- `resolved_relation_fields_v1.tsv`

Core scripts:

- `src/stage3_relation/build_formulation_relation_artifacts_v1.py`

Supporting stage-local wrapper:

- `src/stage3_relation/run_formulation_relation_artifacts_v1.py`

### Stage 4

Stage 4 is the candidate-instance diagnostic layer.

Core script:

- `src/stage4_eval/eval_weak_labels_v7pilot3.py`

Supporting review surface:

- `src/stage4_eval/build_dev15_review_workbook_v1.py`

Important:

- Stage 4 outputs are diagnostic and reviewer-facing.
- They are not the production endpoint and they are not the benchmark-valid
  system result.

### Stage 5

Stage 5 is the only active final-output and benchmark-comparison namespace.

Core scripts:

- `src/stage5_benchmark/build_minimal_final_output_v1.py`
- `src/stage5_benchmark/compare_final_table_to_gt_v1.py`

Supporting stage-local helper:

- `src/stage5_benchmark/run_minimal_final_output_v1.py`
- `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py`

Wrapper status:

- `src/stage5_benchmark/run_minimal_final_output_v1.py` is
  `NON-CANONICAL, STAGE5_ONLY`
- keep it only as a convenience wrapper for Stage 5A closure
- do not use it to imply hidden orchestration or complete-pipeline execution

Completion artifacts:

- `final_formulation_table_v1.tsv`
- `final_table_vs_gt_counts.tsv`
- `final_table_vs_gt_summary.md`

Required Stage 3 inputs:

- `formulation_relation_records_v1.tsv`
- `resolved_relation_fields_v1.tsv`

Optional Layer 2 GT review export:

- `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py`
- This helper builds a run-scoped XLSX boundary-GT workbook from the Stage 5 final table and optional provenance artifacts.
- It is a reviewer-facing support surface, not a production-path completion artifact.

Materialization rule:

- Stage 5 materializes direct extraction fields and explicit Stage 3 resolved
  relation fields only.
- Stage 5 must not perform semantic inference, donor search, or silent
  relation-layer bypass.

Production-path endpoint:

- `final_formulation_table_v1.tsv`

Comparison-node inputs:

- `final_formulation_table_v1.tsv`
- fixed manual GT workbook
- declared scope manifest TSV

## Evaluation Reference Path

Manual GT belongs here, not in the production path.

Reference inputs:

- `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v1_fixed.xlsx`
  - preserved prior authority: `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v1_fixed.xlsx`
  - current authority: `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v2_variantaware.xlsx`
- other checked manual assets under `data/cleaned/labels/manual/` when needed

Rule:

- manual GT is an external evaluation reference used only at the comparison
  node
- it is not a system-produced pipeline stage

## Run Discipline

Every `data/results/run_*` directory must include a reproducibility-grade
`RUN_CONTEXT.md`.

Required run classification:

- `intermediate_diagnostic_run`
- `component_regression_run`
- `full_pipeline_benchmark_run`

Hard rule:

- only `full_pipeline_benchmark_run` may report official GT comparison results

Corollary:

- Stage 2 outputs may be compared to GT only for diagnosis
- Stage 4 outputs may be compared to GT only for diagnosis
- the reported system result must come from Stage 5 final-table comparison only
- the fixed manual GT workbook is a reference input to the comparison node, not
  a production-stage transformation artifact

## Run-Lineage Discipline

Top-level `data/results/run_*` directories are reserved for independent
benchmark or experiment lineages.

Use a child execution under an existing lineage when the work is any of the
following:

- a retry for one or more failed papers
- a partial rerun to complete an interrupted lineage
- a deterministic repair or refresh step
- a stage-only child execution such as Stage 3 materialization for the same
  parent benchmark
- a deterministic merge or completion step that still belongs to the same
  declared lineage objective

Recommended child placement:

- `data/results/<parent_run_id>/lineage/children/<ordered_role>/<child_run_id>/`

Required parent-lineage artifacts when child runs exist:

- parent `RUN_CONTEXT.md`
- child-step mapping or index under `lineage/`
- explicit notes in the parent run flow when child paths were nested after the
  original execution

Do not create multiple sibling top-level run directories that differ only by
retry, remaining, refresh, complete, or stage suffix when they belong to the
same lineage.

## Optional Diagnostic / Review Path

The optional diagnostic/review path consists of:

- `src/stage4_eval/eval_weak_labels_v7pilot3.py`
- `src/stage4_eval/build_dev15_review_workbook_v1.py`
- `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py`

This path is downstream of Stage 2 candidate extraction and exists for mismatch
analysis, review, and debugging. It is not the production endpoint.

## Incremental Reuse Discipline

Reuse is allowed only when the reused artifact is:

- unchanged
- declared in the run context
- traceable to the producing script and run

Typical valid reuse patterns:

- reuse `data/raw/zotero/zotero_selected_items.jsonl`
- reuse `data/cleaned/index/manifest_current.tsv`
- reuse cleaned text or table assets
- reuse a Stage 3 relation-record TSV while iterating only on Stage 5 closure
- reuse a Stage 2 candidate TSV while iterating on Stage 5 closure logic
- reuse the fixed manual GT workbook as a declared comparison input

Typical invalid reuse patterns:

- using an undocumented hand-edited TSV as a stage completion artifact
- mixing artifacts from different declared scopes
- skipping Stage 5 comparison and claiming benchmark validity anyway

## Current Canonical Endpoints

Final structured formulation output:

- `final_formulation_table_v1.tsv`

Comparison-node outputs:

- `final_table_vs_gt_counts.tsv`
- `final_table_vs_gt_summary.md`

The production path yields the final formulation table. The benchmark-valid
result is obtained only when the comparison node reads that table together with
the fixed GT workbook and declared scope manifest as separate inputs.

## What This Runbook Does Not Allow

- no hidden one-click full rerun wrapper
- no duplicate active Stage 5 namespace
- no reporting of Stage 2 or Stage 4 artifacts as final benchmark outputs
- no undocumented shortcut from partial artifacts directly to benchmark claims

## Layer 1 GT Counting Rule

Layer 1 GT counts are formulation-instance counts, not full design-space
counts.

- Count a GT formulation only when the paper reports a realized experimental
  instance with row-level or result-level evidence.
- Do not count methods-only combinations or sweep conditions that were not
  reported as concrete formulation instances.
