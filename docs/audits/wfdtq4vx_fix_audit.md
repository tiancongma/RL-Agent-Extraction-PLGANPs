# WFDTQ4VX Fix Audit

## Scope

This audit reconstructs what the repository explicitly proves about the
supposed WFDTQ4VX "fix" that was discussed before the `2026-04-15`
operational replay.

This is an evidence audit only.

No claim below is made unless it is supported by a maintained code path, a
governed run artifact, or an explicit governance document.

## Short answer

The repository does **not** prove one single clean WFDTQ4VX fix.

Instead, it shows three different WFDTQ4VX-related interventions across
different boundaries:

1. an older **Stage4/compare-style reconciliation rule**
2. a maintained **Stage5 narrow checkpoint-collapse rule**
3. a newer **Stage2 live-boundary recovery effort** that improved current live
   runs but was **not replay-verified from the frozen raw-response boundary**

The operational misunderstanding came from treating these as if they were one
merged end-to-end fix.

## Evidence trail

### 1. Older decision: checkpoint / validation rows should collapse by coordinate signature

Evidence:

- `project/4_DECISIONS_LOG.md`
- `src/stage4_eval/eval_weak_labels_v7pilot3.py`

What the repo explicitly records:

- On `2026-03-10`, the repo recorded the decision:
  - for `WFDTQ4VX`, checkpoint / validation rows should be reconciled by
    factor-level coordinate signature, not table position
  - the resolved interpretation in that decision log was:
    - `Correct formulation-core count = 29`
- The same decision log explicitly says that on `2026-03-10` the validated
  coordinate-signature merge was integrated into:
  - `src/stage4_eval/eval_weak_labels_v7pilot3.py`

What stage it belonged to:

- `compare-only / Stage4 evaluation-style reconciliation`

What input boundary it assumed:

- predicted formulation rows already exist
- the rule acts after row generation, not before it

Merged status:

- merged into code, but only in the historical evaluator path cited above
- not the active Stage2 row-generation boundary

Why this matters:

- this rule can reconcile or collapse **existing** checkpoint rows
- it cannot create the missing WFDTQ4VX row universe when Stage2 emits only
  two benchmark-facing rows

### 2. Maintained Stage5 rule: narrow WFDTQ4VX checkpoint collapse

Evidence:

- `src/stage5_benchmark/build_minimal_final_output_v1.py`
- `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `project/ACTIVE_PIPELINE_FLOW.md`

What the repo explicitly proves:

- the maintained Stage5 builder contains a paper-specific collapse rule:
  - `wfdtq4vx_checkpoint_coordinate_signature_match`
- the builder explicitly states it:
  - applies the validated WFDTQ4VX checkpoint coordinate rule
  - but does **not** implement generalized DOE coordinate reconciliation

What stage it belonged to:

- `Stage5`

What input boundary it assumed:

- `Stage5 closure`
- requires checkpoint / validation candidate rows to already exist upstream in
  the completed Stage2 artifact and final-table candidate set

Merged status:

- merged into maintained code

Important limit:

- this rule is only useful if WFDTQ4VX checkpoint/design rows are already
  present in Stage2/Stage3 inputs
- on the `2026-04-15` replay path, they were not

### 3. Paper-local S2-2 / S2-7 recovery work in the `2026-04-10` repair lineage

Evidence:

- `data/results/20260410_a165cd1/11_s2_dev15_full_freeze_v1/analysis/wfdtq4vx_selector_doe_multiblock_patch_audit_v1.md`
- `data/results/20260410_a165cd1/11_s2_dev15_full_freeze_v1/analysis/wfdtq4vx_s2_7_authorized_target_binding_impl_audit_v1.md`
- `data/results/20260410_a165cd1/11_s2_dev15_full_freeze_v1/analysis/wfdtq4vx_doe_recovery_final_report.md`

What those artifacts show:

- selector-side audit:
  - `Table 15` was added into the formulation-bearing evidence set
  - the paper moved from weak DOE visibility to explicit multi-block coverage
- intermediate S2-7 audit:
  - `authorized_binding_fixed_but_no_row_recovery`
  - explicit result:
    - `s2_7_row_count_after_patch: 4`
- later paper-local final report:
  - claims `31` Stage2/Stage5 rows for WFDTQ4VX
  - claims S2-2 normalized payload repair plus S2-7 target binding repair

What stage it belonged to:

- primarily `Stage2`
  - `S2-2` evidence / table authority handling
  - `S2-7` downstream-ready compatibility projection

What input boundary it assumed:

- earlier than frozen completed Stage2
- specifically `pre-LLM evidence` plus `current Stage2 execution`

Merged status:

- the repo proves these ideas existed in a governed run lineage
- the repo does **not** prove that every paper-local exploratory step in that
  lineage became a replay-frozen maintained baseline

Most important caveat:

- the intermediate audit in the same lineage still showed only `4` rows after
  the authorized-target binding patch
- so the paper-local recovery story was iterative and not one atomic fix

### 4. `2026-04-14` maintained live-boundary claim

Evidence:

- `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `docs/audits/repair_track_status_2026-04-14.md`
- `data/results/20260414_0011ee7/05_s2_4b_live_llm_call/analysis/s2_4b_request_summary_v1.tsv`
- `data/results/20260414_0011ee7/06_s2_4b_live_llm_call_retry/analysis/s2_4b_request_summary_v1.tsv`
- `data/results/20260414_0011ee7/11_s2_4b_wfdtq4vx_persistence_validation/analysis/s2_4b_request_summary_v1.tsv`
- `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/RUN_CONTEXT.md`
- `data/results/20260414_0011ee7/32_full_dev15_rebuilt_from_current_stage2_v1/analysis/final_table_vs_gt_counts.tsv`

What the repo explicitly proves:

- the runbook records:
  - `S2-4b call-layer persistence hardening`
  - `recovered WFDTQ4VX at the maintained live-call boundary`
- the request summaries show:
  - `05_s2_4b_live_llm_call`: WFDTQ4VX request failure
  - `06_s2_4b_live_llm_call_retry`: WFDTQ4VX request failure again
  - `11_s2_4b_wfdtq4vx_persistence_validation`: success with
    `raw_payload_persisted = yes`
- the full rebuilt live-current Stage2 run
  - `32_full_dev15_rebuilt_from_current_stage2_v1`
  - used `source_mode = live_llm`
  - produced `WFDTQ4VX final_table_count = 33`

What stage it belonged to:

- primarily `Stage2`
- specifically the `S2-4b -> S2-5 -> S2-7` live-call path

What input boundary it assumed:

- `raw LLM response at the current maintained live-call boundary`
- not the older frozen raw-response baseline used in today's replay

Merged status:

- partly merged into maintained code and maintained runbook contract
- but not proven replay-verified from the older saved raw-response freeze

## What yesterday's "fix" actually was

The strongest repo-supported interpretation is:

- yesterday's apparent WFDTQ4VX recovery was **primarily a Stage2 live-boundary
  recovery claim**
- it was supported by:
  - maintained S2-4b persistence hardening
  - a fresh current Stage2 composite run from live LLM outputs
  - a resulting full downstream run with `33` WFDTQ4VX final rows

That is **not** the same thing as:

- a replay-verified fix from the old frozen raw-response boundary
- or a benchmark-verified final solution

## Stage classification

The repo evidence supports this split:

- primary fix class:
  - `Stage2`
- secondary older rule:
  - `Stage5`
- older historical reconciliation support:
  - `compare-only / Stage4 evaluator`

## Merge / locality classification

| intervention | stage | merged into maintained code? | only local run evidence? | only docs? | only validator/compare? |
| --- | --- | --- | --- | --- | --- |
| coordinate-signature reconciliation in `eval_weak_labels_v7pilot3.py` | compare-only / Stage4 evaluator | yes | no | no | yes |
| narrow WFDTQ4VX checkpoint rule in `build_minimal_final_output_v1.py` | Stage5 | yes | no | no | no |
| S2-4b persistence hardening | Stage2 live-call boundary | yes, at least partly | no | also documented | no |
| `20260410` selector / authorized-target / normalized-payload paper-local audits | Stage2 | not fully proven as one merged replay contract | yes | no | no |
| `20260414` live-current full rebuilt run with `33` rows | Stage2 -> Stage5 downstream lineage | run evidence only | yes | no | no |

## Audit conclusion

The repository does **not** prove that WFDTQ4VX had a single replay-safe,
benchmark-safe fix as of yesterday.

It proves instead that:

- a narrow Stage5 checkpoint rule exists
- an older evaluator reconciliation rule exists
- a live-current Stage2 lineage on `2026-04-14` recovered WFDTQ4VX strongly
- but the replay boundary used today was still the older frozen raw-response
  lineage, and that lineage never carried the same recovery forward

So the supposed fix was real only in a narrower sense:

- `current maintained live-boundary recovery`

It was **not** proven as:

- `frozen-raw replay recovery`
- `benchmark-verified recovery`
