# WFDTQ4VX Repeat Failure Analysis

## Scope

This audit explains why the `2026-04-15` operational no-LLM replay still
failed for `WFDTQ4VX`.

Replay path under review:

- frozen raw responses
- `S2-5 semantic parsing`
- `S2-6 contract validation`
- `S2-7 compatibility projection`
- `Stage3`
- `Stage5`

## Current replay facts

### Today's replay result

Evidence:

- `data/results/20260415_4f1c2ab/03_s2_7_completed_stage2/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- `data/results/20260415_4f1c2ab/04_operational_baseline_final/final_formulation_table_v1.tsv`
- `data/results/20260415_4f1c2ab/04_operational_baseline_final/paper_level_formulation_count_comparison.tsv`
- `data/results/20260415_4f1c2ab/04_operational_baseline_final/identity_freeze_summary_v1.tsv`

Observed:

- Stage2 completed artifact contains only `3` WFDTQ4VX rows:
  - `FC1_Lopinavir_PLGA_NPs_General`
  - `FC2_Coumarin_PLGA_NPs`
  - `FC3_Plain_Drug_Suspension`
- Stage5 final table keeps only `2` benchmark-facing WFDTQ4VX rows
- final count result:
  - predicted `2`
  - GT `30`
  - difference `-28`

### Today's replay used the old frozen raw boundary

Evidence:

- `data/results/20260415_4f1c2ab/01_s2_5_semantic_parsing_replay/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`

Observed:

- `source_raw_response_path` points to:
  - `data/results/20260402_5c1e7a4/01_dev15_raw_llm_freeze/frozen_raw_responses/WFDTQ4VX__stage2_v2_raw_response.json`

This proves today's replay used the older frozen raw-response lineage.

### The same low-row behavior already existed in the replay-ready lineage

Evidence:

- `data/results/20260414_0011ee7/14_full_pipeline_patched_stage2_dev15_v1/final_formulation_table_v1.tsv`
- `data/results/20260414_0011ee7/14_full_pipeline_patched_stage2_dev15_v1/analysis/final_table_vs_gt_counts_diagnostic_after_identity_freeze_v1.tsv`

Observed:

- that `2026-04-14` downstream run also kept only `2` benchmark-facing
  WFDTQ4VX rows

So today's failure is not a fresh regression inside Stage5.
It is a replay of an already under-recovered upstream lineage.

### The contrasting "good" lineage was different

Evidence:

- `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/RUN_CONTEXT.md`
- `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/final_formulation_table_v1.tsv`
- `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/analysis/final_table_vs_gt_counts.tsv`

Observed:

- that run used:
  - `source_mode = live_llm`
- it produced:
  - `WFDTQ4VX final_table_count = 33`
- its retained WFDTQ4VX rows are mostly batch-level `F_T2_B*` rows from the
  live current Stage2 candidate universe

This is the key contrast:

- current live Stage2 lineage: `33`
- frozen raw replay lineage: `2`

## Why Stage5 did not rescue today's replay

Evidence:

- `src/stage5_benchmark/build_minimal_final_output_v1.py`
- `data/results/20260415_4f1c2ab/04_operational_baseline_final/final_formulation_table_v1.tsv`

Observed:

- the maintained Stage5 builder contains a narrow WFDTQ4VX checkpoint
  coordinate-collapse rule
- but today's final table only received two benchmark-facing candidate rows
- there were no batch-level checkpoint/design rows for Stage5 to collapse,
  reconcile, or retain

Conclusion:

- today's failure was already upstream of Stage5

## Hypothesis testing

### A. The fix was never actually merged into maintained code

Evidence for:

- several paper-specific recovery artifacts are local run analyses:
  - `wfdtq4vx_selector_doe_multiblock_patch_audit_v1.md`
  - `wfdtq4vx_s2_7_authorized_target_binding_impl_audit_v1.md`
  - `wfdtq4vx_doe_recovery_final_report.md`
- today's replay still behaved like the older under-recovered lineage

Evidence against:

- maintained Stage5 code contains the narrow WFDTQ4VX checkpoint rule
- maintained runbook records `S2-4b call-layer persistence hardening`
- maintained live-current Stage2 run `32_full_dev15_rebuilt_from_current_stage2_v1`
  produced `33` rows using maintained entrypoints

Assessment:

- partially true for some paper-local experiments
- not true as a blanket statement

Confidence:

- `medium`

### B. The fix was merged, but not in the path consumed by today's replay

Evidence for:

- very strong lineage split:
  - today's replay uses old frozen raw responses from `2026-04-02`
  - yesterday's stronger WFDTQ4VX recovery came from a `live_llm` current
    Stage2 rebuild on `2026-04-14`
- the runbook itself says WFDTQ4VX was recovered at the maintained
  `live-call boundary`
- the replay-ready `2026-04-14` downstream run built from the replay lineage
  still had only `2` WFDTQ4VX final rows

Evidence against:

- if the fix were entirely downstream and deterministic after raw freeze,
  replay should have inherited it
- it did not

Assessment:

- strongly supported

Confidence:

- `high`

### C. The fix depended on an earlier boundary than frozen raw responses

Evidence for:

- today's replay began at `S2-4b` frozen raw responses from `2026-04-02`
- the runbook explicitly describes the successful hardening as
  `S2-4b call-layer persistence hardening`
- the `2026-04-14` paper-specific post-selector marker replay from a newer raw
  payload reached `4` semantic formulation candidates, while the older frozen
  raw replay remained at `3`
- the `2026-04-14` live-current full Stage2 run reached `33`

Evidence against:

- the maintained Stage5 collapse rule exists later than the raw boundary
- but that rule still depends on upstream row generation and therefore does not
  negate the boundary problem

Assessment:

- strongly supported

Confidence:

- `high`

### D. The fix exists only in compare / validator / downstream reconciliation, not in row generation

Evidence for:

- the original `2026-03-10` WFDTQ4VX coordinate-signature rule was explicitly
  integrated into:
  - `src/stage4_eval/eval_weak_labels_v7pilot3.py`
- that path reconciles counts after predicted rows already exist
- it cannot generate missing Stage2 batch rows

Evidence against:

- the `2026-04-14` live-current Stage2 full rebuild produced many WFDTQ4VX
  rows before compare
- so there was also a row-generation-side recovery in a newer path

Assessment:

- true for the older historical rule
- not sufficient to explain the full `2026-04-14` live recovery

Confidence:

- `medium`

### E. The fix was overwritten or bypassed by later changes

Evidence for:

- today's run did not use the `2026-04-14` live-current Stage2 lineage
- it used the replay-governed older frozen raw-response lineage instead
- operationally, this looks like the recovery "disappeared"

Evidence against:

- I found no explicit repo evidence that a later code change reverted the fix
- I found no governed artifact proving the `2026-04-14` live recovery was
  overwritten in code after it existed

Assessment:

- `bypassed by lineage / boundary choice` is supported
- `overwritten by later code` is not proven

Confidence:

- `low` for overwrite
- `high` for bypass-by-lineage

## Root-cause conclusion

The most evidence-supported explanation is:

- **B + C together**

Expanded:

- the apparent WFDTQ4VX recovery existed in a newer current live Stage2 path
- today's replay did **not** start from that recovered boundary
- it started from the older frozen raw-response payload, which still encoded a
  much smaller semantic candidate universe
- once that smaller universe entered `S2-5`, the maintained deterministic
  downstream path had no lawful basis to generate the missing DOE/design rows

## Layer diagnosis

Primary failing layer:

- `Stage2`

More precise statement:

- the failure sits at the boundary between:
  - the saved `S2-4b` raw-response payload
  - and the Stage2 semantic / completed artifact it can lawfully produce

Stage3 role:

- not the primary root cause
- it can only materialize relations for rows that exist

Stage5 role:

- not the primary root cause
- its narrow WFDTQ4VX checkpoint rule was never given a rich enough candidate
  universe to act on

Compare layer role:

- older compare/reconciliation logic exists
- but it does not repair row generation

## Most likely reason the "fix disappeared"

The single most likely reason is:

- **the supposed fix was never frozen and replay-verified at the raw-response
  boundary that today's operational replay consumed**

In practical repo terms:

- yesterday's stronger WFDTQ4VX recovery belonged to a newer live/current
  Stage2 lineage
- today's replay reused an older frozen raw-response lineage
- those are not the same governed input boundary
