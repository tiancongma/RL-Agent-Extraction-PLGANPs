# Minimal Final-Output Layer Integration Plan

## Insertion point in the current active path

The new layer should be inserted after candidate-instance extraction and before any benchmark-valid GT reporting.

Minimum future order:

1. semantic Stage2 object generation
2. deterministic compatibility projection into the legacy wide-row surface
3. optional Stage4 candidate-instance diagnostics
4. minimal final-output layer
5. final-output benchmark comparison
6. review/export surfaces built from the final-output benchmark artifacts

This preserves the current diagnostic path while adding the missing benchmark-valid closure step.

Current implementation status:

- `src/stage5_benchmark/build_minimal_final_output_v1.py` implements the conservative final-output closure layer.
- `src/stage5_benchmark/compare_final_table_to_gt_v1.py` implements the downstream final-table benchmark comparison step.

## Likely upstream dependencies

Primary upstream dependencies:

- `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`
- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
- the run-scoped compatibility-projected candidate TSV the adapter produces
- the split manifest used for the run

Optional upstream dependencies:

- `src/stage4_eval/eval_weak_labels_v7pilot3.py` outputs for diagnostic context only
- cleaned text/table assets if a narrow evidence-backed guardrail needs to inspect source anchors

## Reference-only or adapter-source scripts

Useful reference assets:

- `src/stage5_benchmark/formulation_core_signature_v1.py`
- `src/stage5_benchmark/build_two_table_schema_v2.py`
- `src/stage5_benchmark/build_two_table_schema_v3.py`
- `archive/code/stage5_merge_publish/merge_results.py`

Historical reference-only assets:

- `archive/code/dev15_skeleton_bootstrap/*`

These assets may inform signature choice, schema shape, or export patterns. They do not constitute the missing layer by themselves.

## Minimum viable implementation first

The first implementation should do only four things:

1. ingest candidate-instance rows from the compatibility-projected legacy Stage2 surface
2. remove explicit non-formulation rows
3. apply narrow collapse logic for clearly redundant candidate rows using a decision trace
4. emit a benchmark-valid final formulation table and its decision-trace artifacts

The corresponding benchmark comparison step should then consume that final table.

## What should be deferred

The following should be deferred until after the minimal layer is working:

- broad inheritance repair
- large DoE reconstruction beyond narrow clear cases
- measurement-level schema expansion
- database publication/export restructuring
- historical Stage5 compatibility cleanup

## Run type unlocked by this layer

Because this layer now exists and a benchmark comparison step consumes its final formulation table, the repository can legitimately execute `full_pipeline_benchmark_run` only when the complete manual Stage 0 to Stage 5 path is run for the declared scope.

Outside that complete path:

- current Stage2 -> Stage4 runs remain `intermediate_diagnostic_run` or `component_regression_run`
- direct candidate-row-vs-GT comparison remains diagnostic-only
