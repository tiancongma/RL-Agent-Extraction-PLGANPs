# EE35 Full Loss-Class Repair Execution Plan

## Status
`executed_diagnostic_only`

## Scope
This plan covers the campaign-local EE35 diagnostic lineage under
`data/results/20260511_b069802`.

Goal: explain and repair, where lawful, why GPT Web master has 402 formulation
rows while the new S2-4a/API lineage reached only 155 Stage5 rows before repair
and 212 rows after the R1-R4 S2-7 repairs executed here.

This is diagnostic-only and not benchmark-valid.

## Governing Inputs
- GPT Web master reference:
  - `data/results/20260511_b069802/17_ee35_master_formulation_gt_from_gpt_web/final_table/ee35_master_formulation_gt_final_table_draft_v3.tsv`
- Current API raw responses:
  - `data/results/20260511_b069802/168_stage2_s2_4b_ee35_new_s2_4a_live_same_params/raw_responses`
- Current S2-5 parse:
  - `data/results/20260511_b069802/169_stage2_s2_5_ee35_new_s2_4a_semantic_parse/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
- Current S2-6 validation:
  - `data/results/20260511_b069802/170_stage2_s2_6_ee35_new_s2_4a_validation`
- Latest repaired S2-7 replay:
  - `data/results/20260511_b069802/184_stage2_s2_7_ee35_condition_row_repair`
- Latest repaired Stage3 replay:
  - `data/results/20260511_b069802/185_stage3_ee35_condition_row_repair_relation_artifacts`
- Latest repaired Stage5 replay:
  - `data/results/20260511_b069802/186_stage5_ee35_condition_row_repair_final_output`

## Loss Classes And Repairs

### R1 Native Row Identity Repair
- status: `completed`
- scripts:
  - `src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py`
  - `src/stage2_sampling_labels/table_row_expansion_v1.py`
- repair:
  - accept native prefixed row labels such as `NP 1`, `HbNPs-12`, and `PBD1`
    only inside existing LLM-authorized table/DOE row-expansion paths.
- validation:
  - unit tests pass
  - S2-7 rows improved `216 -> 238`
  - Stage5 rows improved `155 -> 185`

### R2 Partial Table Expansion Must Not Suppress LLM Semantic Summary Rows
- status: `completed`
- script:
  - `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
- repair:
  - suppress LLM summary rows only when table expansion is structurally complete
    enough to replace them, not when a partial table path emits a small subset.
- validation:
  - unit test added for incomplete table-expansion replacement gate
  - replay `178 -> 179 -> 180`
  - Stage5 rows improved `185 -> 202`

### R3 Non-DOE Table Row Expansion Under-Materialization
- status: `completed_bounded`
- script:
  - `src/stage2_sampling_labels/table_row_expansion_v1.py`
- repair:
  - add source-backed prefixed/contextual identity row materialization under
    already LLM-authorized table scopes.
- validation:
  - unit test covers `NS-01 (10 min homogenization)` style row identities
  - replay `181 -> 182 -> 183`
  - Stage5 rows improved `202 -> 206`
- remaining blocker:
  - papers without preserved table payloads, such as `GWWNCC35`, need Stage1/S2-2
    table-asset restoration or governed source-text table reconstruction.

### R4 DOE Or Multi-Variable Condition Matrix Emits No Rows
- status: `completed_bounded_with_residual_prompt_or_asset_losses`
- script:
  - `src/stage2_sampling_labels/table_row_expansion_v1.py`
- repair:
  - add source-backed condition-row materialization for LLM-authorized DOE or
    multi-variable condition matrices.
- validation:
  - unit test confirms grouped factor-level rows such as `Polymer amount: 50 mg`
    are retained while single-variable tables still defer to the established
    recovery path
  - replay `184 -> 185 -> 186`
  - S2-7 rows improved `243 -> 259`; Stage5 rows improved `206 -> 212`
- remaining blocker:
  - current API raw responses authorize far fewer semantic formulation rows than
    the GPT Web master reference. Deterministic replay cannot lawfully create
    missing formulation families when the current raw response marks tables as
    non-formulation or provides no formulation candidates.

### R5 Stage5 Filter Or Identity Closure Loss
- status: `diagnosed_no_global_filter_patch`
- script:
  - `src/stage5_benchmark/build_minimal_final_output_v1.py`
- diagnosis:
  - Stage5 filters 44 rows in run `186`, mostly semantic summaries superseded by
    row-level table enumeration, downstream non-synthesis descendants, or rows
    explicitly marked `candidate_non_formulation`.
  - No global Stage5 weakening is justified from this pass.

### R6 S2-4b Semantic Underselection
- status: `diagnosed_unresolved_without_new_live_call`
- scripts:
  - `src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py`
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
- diagnosis:
  - current API raw responses for `GNQWKY3J`, `TT2JDLQK`, `X6AJ9LPV`, and
    `KTNLRQZU` contain no formulation candidates or mark relevant tables as
    non-formulation, while GPT Web master contributes rows for the same papers.
  - This requires prompt/evidence/live-call repair or lawful replay from a
    stronger frozen raw-response boundary; S2-7/S3/S5 must not invent the missing
    semantic universe.

## Final Diagnostic Outcome
- latest lawful diagnostic chain:
  - S2-7: `184_stage2_s2_7_ee35_condition_row_repair`
  - Stage3: `185_stage3_ee35_condition_row_repair_relation_artifacts`
  - Stage5: `186_stage5_ee35_condition_row_repair_final_output`
- row counts:
  - original post-live Stage5: `155`
  - after R1: `185`
  - after R2: `202`
  - after R3: `206`
  - after R4: `212`
- conclusion:
  - downstream S2-7/S3/S5 did contain real generic bugs, now repaired within the
    lawful LLM-authorized boundary.
  - the remaining gap to GPT Web master `402` is not explainable by one more
    downstream parser/projection fix alone; it is dominated by current S2-4b raw
    underselection and missing/preserved-table-asset lineage gaps for selected
    papers.

## Acceptance Criteria
- Every loss class above has one of:
  - code repair plus bounded replay, or
  - explicit no-repair legality explanation.
- The final audit identifies the responsible stage for remaining losses.
- No downstream script creates formulations without LLM semantic authorization.
