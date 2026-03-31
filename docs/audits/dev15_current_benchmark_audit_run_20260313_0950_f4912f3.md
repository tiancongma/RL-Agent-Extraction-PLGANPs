# DEV15 Current Benchmark Audit

## 1. Objective

- Reuse the audited current 5-paper lineage.
- Run the remaining 10 DEV15 papers under the same current Stage2 workflow standard.
- Merge both current lineages into one provenance-backed DEV15 current result.
- Compare the merged DEV15 current Stage5 final table against the fixed GT workbook at DOI level.

## 2. Reused current 5-paper lineage

- Stage2 anchor:
  - `data/results/run_20260312_1321_455ac37_targeted5_stage2_regression_v1`
- Reused artifacts:
  - `RUN_CONTEXT.md`
  - `targeted_manifest.tsv`
  - `weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`
  - `weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.jsonl`
  - `weak_labels_v7pilot_r3_fixparse/raw_responses/`
- Reused downstream reference lineage:
  - `data/results/run_20260312_1723_455ac37_downstream_only_from_existing_stage2_v1`
  - used only as the current semantic comparison anchor for Stage5-style outputs, not as the merged baseline artifact

## 3. New remaining-10 run lineage

- Stage2 run:
  - `data/results/run_20260313_0950_f4912f3_dev15_remaining10_current_stage2_extraction_v1`
- Stage2 script:
  - `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
- Stage2 scope file:
  - `data/results/run_20260313_0950_f4912f3_dev15_remaining10_current_stage2_extraction_v1/remaining10_scope.tsv`
- Stage2 outputs:
  - `weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`
  - `weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.jsonl`
  - `weak_labels_v7pilot_r3_fixparse/raw_responses/`
- Stage2 result summary:
  - `10` papers
  - `95` candidate rows
  - Stage2 TSV SHA256: `7ABB7AF44AC97682C008A2B4F500EF86244FBDC84E8999746651A2E2267FEAD3`

- Stage5 remaining-10 reuse run:
  - `data/results/run_20260313_0950_f4912f3_dev15_remaining10_current_stage5_closure_v1`
- Stage5 scripts:
  - `src/stage5_benchmark/build_minimal_final_output_v1.py`
  - `src/stage5_benchmark/compare_final_table_to_gt_v1.py`
- Stage5 remaining-10 outputs:
  - `final_formulation_table_v1.tsv`
  - `final_output_decision_trace_v1.tsv`
  - `final_table_vs_gt_counts.tsv`
  - `doi_level_formulation_count_comparison.tsv`
- Remaining-10 Stage5 result summary:
  - `95` input candidate rows
  - `76` final rows
  - `5/10` papers matched GT counts
  - diagnostic-only by contract

## 4. Merged DEV15 current lineage

- Merged benchmark run:
  - `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1`
- Merge helper:
  - `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/merge_stage2_scope_artifacts_v1.py`
- Merged scope:
  - `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/dev15_scope.tsv`
- Merged Stage2 outputs:
  - `weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`
  - `weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.jsonl`
  - `weak_labels_v7pilot_r3_fixparse/raw_responses/`
  - `weak_labels_v7pilot_r3_fixparse/merge_summary.json`
- Merged Stage2 summary:
  - `15` DOI-scoped JSONL records
  - `222` merged candidate rows
  - merged Stage2 TSV SHA256: `98852804B2B4899BA8E1EFE51C68321AB68DD58D5EF92036BFB49BD17022B3AF`

## 5. Exact script sequence and run steps

1. Reused the existing 5-paper Stage2 current artifact from `run_20260312_1321_455ac37_targeted5_stage2_regression_v1`.
2. Ran remaining-10 Stage2 extraction with:
   - `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
3. Ran remaining-10 Stage5 closure and GT comparison with:
   - `src/stage5_benchmark/build_minimal_final_output_v1.py`
   - `src/stage5_benchmark/compare_final_table_to_gt_v1.py`
4. Merged the reused 5-paper and new remaining-10 Stage2 artifacts with:
   - `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/merge_stage2_scope_artifacts_v1.py`
5. Attempted optional merged Stage4 diagnostics with:
   - `src/stage4_eval/eval_weak_labels_v7pilot3.py`
   - result: failed, see `stage4_diagnostics/STAGE4_FAILURE.md`
6. Built merged DEV15 Stage5 final output with:
   - `src/stage5_benchmark/build_minimal_final_output_v1.py`
7. Compared merged DEV15 Stage5 final output to GT with:
   - `src/stage5_benchmark/compare_final_table_to_gt_v1.py`

## 6. Exact manifests and DOI membership

Reused 5-paper keys and DOIs:
- `5GIF3D8W` / `10.1080/10717540802174662`
- `5ZXYABSU` / `10.2147/IJN.S130908`
- `L3H2RS2H` / `10.1016/j.ejpb.2004.09.002`
- `WFDTQ4VX` / `10.1080/10717544.2016.1199605`
- `WIVUCMYG` / `10.1002/jps.24101`

New remaining-10 keys and DOIs:
- `7ZS858NS` / `10.1021/acsomega.0c00111`
- `BB3JUVW7` / `10.1016/j.ijpharm.2021.120820`
- `BXCV5XWB` / `10.1007/s10439-019-02430-x`
- `INMUTV7L` / `10.3390/nano10040720`
- `PA3SPZ28` / `10.1038/s41598-017-00696-6`
- `QLYKLPKT` / `10.2147/IJN.S54040`
- `RHMJWZX8` / `10.1111/jphp.12481`
- `UFXX9WXE` / `10.1155/2014/156010`
- `V99GKZEI` / `10.1039/C5RA27386B`
- `YGA8VQKU` / `10.1016/j.colsurfb.2009.03.028`

Full merged DEV15 scope file:
- `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/dev15_scope.tsv`

## 7. Output files produced

Main remaining-10 outputs:
- `data/results/run_20260313_0950_f4912f3_dev15_remaining10_current_stage2_extraction_v1/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`
- `data/results/run_20260313_0950_f4912f3_dev15_remaining10_current_stage5_closure_v1/final_formulation_table_v1.tsv`
- `data/results/run_20260313_0950_f4912f3_dev15_remaining10_current_stage5_closure_v1/doi_level_formulation_count_comparison.tsv`

Main merged DEV15 outputs:
- `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`
- `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/final_formulation_table_v1.tsv`
- `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/final_table_vs_gt_counts.tsv`
- `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/doi_level_formulation_count_comparison.tsv`
- `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/MERGE_SUMMARY.md`
- `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/RUN_CONTEXT.md`

## 8. GT comparison summary

Merged DEV15 current benchmark result:
- `200` predicted final formulation rows
- `207` GT formulation rows
- `7/15` papers matched GT counts
- `8/15` papers mismatched GT counts
- final table SHA256:
  - `684830D051ACE3C0F1728EFD859EC3F5FF1A60F6D5C9B2D0EAACD8E48B3621DF`

Per-paper DOI-level summary:

| doi | paper_key | gt_formulation_count | predicted_formulation_count | count_delta | counts_match |
|---|---|---:|---:|---:|---|
| 10.1080/10717540802174662 | 5GIF3D8W | 32 | 38 | 6 | no |
| 10.2147/IJN.S130908 | 5ZXYABSU | 9 | 9 | 0 | yes |
| 10.1021/acsomega.0c00111 | 7ZS858NS | 1 | 1 | 0 | yes |
| 10.1016/j.ijpharm.2021.120820 | BB3JUVW7 | 12 | 12 | 0 | yes |
| 10.1007/s10439-019-02430-x | BXCV5XWB | 3 | 9 | 6 | no |
| 10.3390/nano10040720 | INMUTV7L | 12 | 12 | 0 | yes |
| 10.1016/j.ejpb.2004.09.002 | L3H2RS2H | 22 | 21 | -1 | no |
| 10.1038/s41598-017-00696-6 | PA3SPZ28 | 5 | 5 | 0 | yes |
| 10.2147/IJN.S54040 | QLYKLPKT | 7 | 8 | 1 | no |
| 10.1111/jphp.12481 | RHMJWZX8 | 1 | 2 | 1 | no |
| 10.1155/2014/156010 | UFXX9WXE | 26 | 4 | -22 | no |
| 10.1039/C5RA27386B | V99GKZEI | 6 | 6 | 0 | yes |
| 10.1080/10717544.2016.1199605 | WFDTQ4VX | 29 | 30 | 1 | no |
| 10.1002/jps.24101 | WIVUCMYG | 26 | 26 | 0 | yes |
| 10.1016/j.colsurfb.2009.03.028 | YGA8VQKU | 16 | 17 | 1 | no |

## 9. DOI-level list of papers needing manual inspection

- `5GIF3D8W` / `10.1080/10717540802174662`: over by `6`
- `BXCV5XWB` / `10.1007/s10439-019-02430-x`: over by `6`
- `L3H2RS2H` / `10.1016/j.ejpb.2004.09.002`: under by `1`
- `QLYKLPKT` / `10.2147/IJN.S54040`: over by `1`
- `RHMJWZX8` / `10.1111/jphp.12481`: over by `1`
- `UFXX9WXE` / `10.1155/2014/156010`: under by `22`
- `WFDTQ4VX` / `10.1080/10717544.2016.1199605`: over by `1`
- `YGA8VQKU` / `10.1016/j.colsurfb.2009.03.028`: over by `1`

## 10. Any unproven or non-compliant points

- The optional merged Stage4 diagnostic path is unproven for the current merged DEV15 artifact because `src/stage4_eval/eval_weak_labels_v7pilot3.py` failed inside the `WFDTQ4VX` checkpoint reconciliation branch.
- The benchmark-valid primary result does not depend on that Stage4 path.
- The reused 5-paper upstream lineage remains `component_regression_run` by its original run contract, but its Stage2 artifacts were explicitly reused and documented inside the merged full-scope benchmark run, which is consistent with the active governance allowance for valid incremental reuse.
