# Final Table vs GT Summary

## Declared scope

- scope_name: `dev15_diagnostic_table_recovery_closure`
- scope_manifest_tsv: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv`
- final_formulation_table_tsv: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f/427_stage5_table_recovery_closure_diagnostic/final_formulation_table_v1.tsv`
- gt_counts_tsv: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/cleaned/gt_authority/v1/dev15_layer1_gt_counts.tsv`

## Compare Contract

- compare_mode: `diagnostic`
- benchmark_mode: `disabled`
- benchmark_valid: `no`

## Diagnostic-phase statement

- This comparison is diagnostic-only in the current development phase.
- The comparison object is the completed Stage 5 final formulation table, not an intermediate artifact.
- No intermediate Stage2 or other partial-layer artifacts are used as the evaluation object, and this output is not benchmark-valid evidence.

## Supported benchmark views

- per-DOI final-formulation count comparison: supported
- EE subset comparison: not supported by the current authoritative GT artifact

## Aggregate outcome

- total_final_table_rows: `200`
- total_gt_rows: `202`
- matched_papers: `13`
- mismatched_papers: `2`

## Per-paper counts

- `5GIF3D8W`: final=`26` gt=`26` diff=`0` status=`match`
- `5ZXYABSU`: final=`9` gt=`9` diff=`0` status=`match`
- `7ZS858NS`: final=`1` gt=`1` diff=`0` status=`match`
- `BB3JUVW7`: final=`12` gt=`12` diff=`0` status=`match`
- `BXCV5XWB`: final=`3` gt=`3` diff=`0` status=`match`
- `INMUTV7L`: final=`12` gt=`12` diff=`0` status=`match`
- `L3H2RS2H`: final=`21` gt=`21` diff=`0` status=`match`
- `PA3SPZ28`: final=`3` gt=`3` diff=`0` status=`match`
- `QLYKLPKT`: final=`7` gt=`7` diff=`0` status=`match`
- `RHMJWZX8`: final=`1` gt=`2` diff=`-1` status=`under`
- `UFXX9WXE`: final=`26` gt=`27` diff=`-1` status=`under`
- `V99GKZEI`: final=`6` gt=`6` diff=`0` status=`match`
- `WFDTQ4VX`: final=`30` gt=`30` diff=`0` status=`match`
- `WIVUCMYG`: final=`26` gt=`26` diff=`0` status=`match`
- `YGA8VQKU`: final=`17` gt=`17` diff=`0` status=`match`

## Limitations

- The current comparison is limited to final-formulation counts for this declared scope.
- The authoritative fixed DEV15 skeleton workbook does not expose structured EE ground-truth fields, so no benchmark-valid EE subset comparison is emitted in this first full-pipeline run.
- Any mismatch investigation must start from these final-table results and only then trace backward into Stage 5 decision-trace artifacts or Stage 2 candidate rows.
