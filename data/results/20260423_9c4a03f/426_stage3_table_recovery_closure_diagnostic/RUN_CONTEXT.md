# RUN_CONTEXT — 426_stage3_table_recovery_closure_diagnostic

- generated_at: 2026-05-10T11:30:12Z
- run_type: diagnostic-only Stage3 replay
- benchmark_valid: false
- purpose: Materialize Stage3 relation artifacts from the repaired S2-7 table-row/single-variable recovery closure lineage.

## Inputs

- Stage2 compatibility TSV: `data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- Stage2 compatibility JSONL: `data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.jsonl`

## Command

```bash
PYTHONPATH=. python3 src/stage3_relation/build_formulation_relation_artifacts_v1.py \
  --weak-labels-tsv data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv \
  --weak-labels-jsonl data/results/20260423_9c4a03f/425_stage2_s2_7_table_recovery_closure_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.jsonl \
  --out-dir data/results/20260423_9c4a03f/426_stage3_table_recovery_closure_diagnostic
```

## Outputs

- `formulation_relation_records_v1.tsv`
- `resolved_relation_fields_v1.tsv`
- `formulation_logic_graph_v1.jsonl`
- `formulation_relation_summary_v1.tsv`

## Summary

- paper_count: 15
- candidate_count: 228
- relation_row_count: 2102
- resolved_relation_field_row_count: 1223

## Lineage note

This is a lawful downstream replay from completed S2-7 artifact 425 into Stage3. It is diagnostic-only and not benchmark-valid.
