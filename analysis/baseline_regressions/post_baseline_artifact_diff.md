# Artifact Difference Audit

## Comparison inputs

- Baseline run:
  - `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1`
- Current lineage:
  - `data/results/20260418_9538ec2`
- Current completed Stage2 source used here:
  - `data/results/20260418_9538ec2/32_diagnosis_restart_s2_7_v1/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- Current downstream compare surface used here:
  - `data/results/20260418_9538ec2/28_diagnosis_baseline_restart_stepwise_v1`

## Structural differences

- `final_formulation_table_v1.tsv`
  - baseline fields: `166`
  - current fields: `166`
  - schema change observed: `no`

- Stage2 completed weak-label TSV
  - baseline fields: `140`
  - current fields: `140`
  - schema change observed: `no`

- Stage3 relation artifacts
  - `formulation_relation_records_v1.tsv`: same field set
  - `resolved_relation_fields_v1.tsv`: same field set

Conclusion:

- The main artifact change is not schema drift.
- The change is row membership and row distribution.

## Row count differences

### Stage2 completed artifact

- Baseline:
  - `108` rows across `14` papers
- Current:
  - `104` rows across `10` papers

### Stage3 relation artifacts

- Baseline `formulation_relation_records_v1.tsv`:
  - `935` rows across `14` papers
- Current `formulation_relation_records_v1.tsv` under `28_...`:
  - `1136` rows across `10` papers

- Baseline `resolved_relation_fields_v1.tsv`:
  - `239` rows across `12` papers
- Current `resolved_relation_fields_v1.tsv` under `28_...`:
  - `320` rows across `10` papers

Interpretation:

- Current Stage3-style outputs are larger despite covering fewer papers.
- Inflation is concentrated in a smaller paper set rather than broad coverage.

### Final formulation table

- Baseline:
  - `93` rows across `14` papers
- Current:
  - `82` rows across `10` papers

## Per-paper row differences

| paper_key | baseline_stage2 | current_stage2 | baseline_final | current_final |
|---|---:|---:|---:|---:|
| 5GIF3D8W | 3 | 0 | 3 | 0 |
| 7ZS858NS | 6 | 7 | 1 | 1 |
| BB3JUVW7 | 3 | 12 | 3 | 12 |
| BXCV5XWB | 3 | 0 | 3 | 0 |
| INMUTV7L | 12 | 24 | 12 | 12 |
| L3H2RS2H | 5 | 3 | 5 | 3 |
| PA3SPZ28 | 3 | 0 | 2 | 0 |
| QLYKLPKT | 3 | 5 | 3 | 4 |
| RHMJWZX8 | 2 | 2 | 2 | 2 |
| UFXX9WXE | 1 | 17 | 1 | 17 |
| V99GKZEI | 13 | 6 | 7 | 6 |
| WFDTQ4VX | 27 | 2 | 27 | 2 |
| WIVUCMYG | 9 | 26 | 6 | 23 |
| YGA8VQKU | 18 | 0 | 18 | 0 |

## Missing papers in current outputs

- Present in baseline final table, absent in current final table:
  - `5GIF3D8W`
  - `BXCV5XWB`
  - `PA3SPZ28`
  - `YGA8VQKU`

- Present in baseline Stage2, absent in current completed Stage2:
  - same four papers above

Interpretation:

- This missing-paper set aligns with the incomplete fresh-live restart, not only with row-boundary changes.

## Largest paper-level shifts

- Strong negative regressions:
  - `WFDTQ4VX`: final `27 -> 2`
  - `YGA8VQKU`: final `18 -> 0`
  - `5GIF3D8W`: final `3 -> 0`
  - `BXCV5XWB`: final `3 -> 0`
  - `PA3SPZ28`: final `2 -> 0`

- Strong increases / inflation:
  - `UFXX9WXE`: final `1 -> 17`
  - `WIVUCMYG`: final `6 -> 23`
  - `BB3JUVW7`: final `3 -> 12`

- Mild improvements:
  - `QLYKLPKT`: final `3 -> 4`

## GT-facing count differences from the compare artifacts

Using:

- baseline compare:
  - `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1/gt_authority_v2_variantaware/final_table_vs_gt_counts.tsv`
- current compare:
  - `data/results/20260418_9538ec2/28_diagnosis_baseline_restart_stepwise_v1/final_table_vs_gt_counts.tsv`
- GT authority:
  - `data/cleaned/gt_authority/v1/dev15_layer1_gt_counts.tsv`

Largest GT-facing changes:

- Worse:
  - `WFDTQ4VX`: `27 -> 2` against GT `30`
  - `YGA8VQKU`: `18 -> 0` against GT `17`
  - `5GIF3D8W`: `3 -> 0` against GT `26`

- Better:
  - `BB3JUVW7`: `3 -> 12` against GT `12`
  - `UFXX9WXE`: `1 -> 17` against GT `27`
  - `WIVUCMYG`: `6 -> 23` against GT `26`
  - `V99GKZEI`: `7 -> 6` against GT `6`

## Root artifact interpretation

- There is no artifact schema break.
- The current lineage differs from baseline in two separable ways:
  - incomplete fresh-live coverage removed four papers entirely
  - selector/prompt changes materially redistributed rows inside the surviving papers

## Caveat

- These differences are diagnostic-only and should be peer-reviewed before action.
