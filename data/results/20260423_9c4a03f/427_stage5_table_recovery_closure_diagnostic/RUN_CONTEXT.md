# RUN_CONTEXT — 427_stage5_table_recovery_closure_diagnostic

- generated_at: 2026-05-10T11:30:12Z
- run_type: diagnostic-only Stage5 replay
- benchmark_valid: false
- purpose: Build Stage5 final formulation table from repaired S2-7 artifact 425 and Stage3 relation artifacts 426.

## Inputs

- Stage2 compatibility TSV: `data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- Stage3 relation records TSV: `data/results/20260423_9c4a03f/426_stage3_table_recovery_closure_diagnostic/formulation_relation_records_v1.tsv`
- Stage3 resolved relation fields TSV: `data/results/20260423_9c4a03f/426_stage3_table_recovery_closure_diagnostic/resolved_relation_fields_v1.tsv`

## Command

```bash
PYTHONPATH=. python3 src/stage5_benchmark/build_minimal_final_output_v1.py \
  --input-tsv data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv \
  --relation-records-tsv data/results/20260423_9c4a03f/426_stage3_table_recovery_closure_diagnostic/formulation_relation_records_v1.tsv \
  --resolved-relation-fields-tsv data/results/20260423_9c4a03f/426_stage3_table_recovery_closure_diagnostic/resolved_relation_fields_v1.tsv \
  --out-dir data/results/20260423_9c4a03f/427_stage5_table_recovery_closure_diagnostic
```

## Outputs

- `final_formulation_table_v1.tsv`
- `downstream_variant_records_v1.tsv`
- `final_output_decision_trace_v1.tsv`
- `final_output_summary_v1.md`

## Summary

- input_rows: 228
- final_rows: 200
- filtered_rows: 27
- collapsed_rows: 1
- downstream_variant_rows: 1

## Lineage note

This is the Stage5 terminal surface consumed by diagnostic compare run 428. It is diagnostic-only and not benchmark-valid.
