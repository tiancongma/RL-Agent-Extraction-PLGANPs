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

Completion artifact:

- `data/results/<stage2_run_id>/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`

### Stage 3

Stage 3 is the formulation consolidation / normalization boundary.

Runtime rule:

- Stage 3 belongs to the production path
- it converts Stage 2 candidate formulation-instance rows into normalized
  formulation records suitable for Stage 5 closure
- the repository does not yet contain a dedicated standalone Stage 3 script
- until such a script exists, Stage 3 must still be recorded as an explicit
  production-boundary contract in provenance and governance docs

Current production-boundary input:

- `data/results/<stage2_run_id>/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`

Current production-boundary output:

- normalized or consolidated formulation records consumed by Stage 5

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

Wrapper status:

- `src/stage5_benchmark/run_minimal_final_output_v1.py` is
  `NON-CANONICAL, STAGE5_ONLY`
- keep it only as a convenience wrapper for Stage 5A closure
- do not use it to imply hidden orchestration or complete-pipeline execution

Completion artifacts:

- `final_formulation_table_v1.tsv`
- `final_table_vs_gt_counts.tsv`
- `final_table_vs_gt_summary.md`

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

## Optional Diagnostic / Review Path

The optional diagnostic/review path consists of:

- `src/stage4_eval/eval_weak_labels_v7pilot3.py`
- `src/stage4_eval/build_dev15_review_workbook_v1.py`

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
