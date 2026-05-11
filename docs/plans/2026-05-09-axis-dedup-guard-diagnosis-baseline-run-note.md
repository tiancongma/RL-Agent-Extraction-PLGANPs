# 2026-05-09 Axis-Dedup Guard Diagnosis Baseline Run Note

## Scope

This note records the execution outcome for the frozen S2-4b diagnosis-baseline repair plan:

- plan: `docs/plans/2026-05-09-frozen-s2-4b-diagnosis-baseline-repair-plan.md`
- progress ledger: `docs/plans/2026-05-09-frozen-s2-4b-diagnosis-baseline-repair-progress.tsv`
- code repair: `src/stage2_sampling_labels/table_row_expansion_v1.py`
- regression tests: `tests/test_stage2_table_row_expansion_scope_alias_roles_v1.py`

All runs below are diagnosis runs. The S2-7 replay reused frozen upstream semantic artifacts and did not make a live LLM call.

## Root cause and guard

`QLYKLPKT` over-count came from supplemental single-variable recovery emitting a `PLGA:ITZ ratio=5:1` row even though row-level table authority had already materialized the same axis through Table 9. The existing table row value was noisy (`surfactant concentration optimized | PLGA:ITZ ratios 5:1`), so exact-value duplicate suppression did not catch it.

The generic guard now:

1. reads already materialized assignment values for the same semantic scope;
2. compares supplemental single-variable levels against exact or contained row-level axis values;
3. suppresses only overlapping supplemental levels, preserving non-overlapping source-supported recovery such as the `5GIF3D8W` drug endpoint rows.

## Test verification

Command:

```bash
PYTHONPATH=. python3 -m unittest \
  tests.test_stage2_table_row_expansion_scope_alias_roles_v1 \
  tests.test_stage2_table_summary_numeric_visibility_v1 -v
```

Result:

```text
Ran 33 tests ... OK
```

## Replay lineage

### S2-7

```text
data/results/20260423_9c4a03f/411_stage2_s2_7_axis_dedup_guard_diagnosis_baseline
```

Completed Stage2 artifact:

```text
data/results/20260423_9c4a03f/411_stage2_s2_7_axis_dedup_guard_diagnosis_baseline/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv
```

Counts checked at S2-7:

```text
total rows: 239
5GIF3D8W: 27
QLYKLPKT: 9
BB3JUVW7: 14
L3H2RS2H: 23
```

### Stage3

```text
data/results/20260423_9c4a03f/412_stage3_axis_dedup_guard_diagnosis_baseline
```

Summary:

```text
candidate_count: 239
relation_row_count: 2292
resolved_relation_field_row_count: 1183
```

### Stage5

```text
data/results/20260423_9c4a03f/413_stage5_axis_dedup_guard_diagnosis_baseline
```

Final table:

```text
data/results/20260423_9c4a03f/413_stage5_axis_dedup_guard_diagnosis_baseline/final_formulation_table_v1.tsv
```

Stage5 counts:

```text
final_rows: 194
5GIF3D8W: 26
QLYKLPKT: 7
BB3JUVW7: 12
L3H2RS2H: 21
```

### Final table vs GT diagnosis comparison

```text
data/results/20260423_9c4a03f/414_compare_axis_dedup_guard_diagnosis_baseline
```

Main DOI count audit:

```text
data/results/20260423_9c4a03f/414_compare_axis_dedup_guard_diagnosis_baseline/final_table_vs_gt_counts_by_doi.tsv
```

Key repaired papers:

```text
5GIF3D8W: GT 26, pred 26, delta 0, match
QLYKLPKT: GT 7, pred 7, delta 0, match
BB3JUVW7: GT 12, pred 12, delta 0, match
L3H2RS2H: GT 21, pred 21, delta 0, match
```

Overall diagnosis comparison summary:

```text
papers_in_scope: 15
papers_matching: 12
papers_mismatching: 3
total_final_table_rows: 194
total_gt_rows: 202
```

## Promotion note

This lineage is suitable for review as the next system-registered diagnosis baseline candidate. I did not update `data/results/ACTIVE_RUN.json` in this execution step.
