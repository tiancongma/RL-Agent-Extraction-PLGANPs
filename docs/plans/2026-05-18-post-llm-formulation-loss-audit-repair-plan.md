# 2026-05-18 Post-LLM Formulation Loss Audit And Repair Plan

## Scope

Campaign: `data/results/20260511_b069802`

Current post-live lineage:

- S2-4a prompts:
  `105_stage2_s2_4a_multi_anchor_floor_locator_campaign_no_live`
- complete S2-4b raw-response freeze:
  `108_stage2_s2_4b_deepseek_multi_anchor_complete_raw_response_freeze`
- S2-5 parse:
  `111_stage2_s2_5_deepseek_multi_anchor_context_marker_filter_parse`
- S2-6 validation:
  `112_stage2_s2_6_deepseek_multi_anchor_context_marker_filter_validation`
- S2-7 completed Stage2:
  `113_stage2_s2_7_deepseek_multi_anchor_complete_projection`
- Stage3:
  `114_stage3_deepseek_multi_anchor_complete_relation`
- Stage5:
  `115_stage5_deepseek_multi_anchor_complete_final_table`

This plan starts after the pre-LLM evidence and prompt-readiness work. Its goal
is to determine whether formulation rows are lost after S2-4b and to repair the
first generic post-LLM function that loses already-authorized formulation
semantics.

## Non-Negotiable Boundary Rules

- Do not make new live LLM calls during this audit.
- Do not patch individual papers.
- Do not use Stage2 raw responses, S2-5 objects, or S2-7 rows as benchmark
  evidence.
- Treat Stage5 child `115` as diagnostic-only, not benchmark-valid.
- Preserve LLM semantic authority: deterministic code may retain, parse,
  reattach, validate, or materialize LLM-authorized semantics, but must not
  invent formulation membership.
- If old lineage rows were produced by overly broad deterministic expansion,
  do not restore that behavior unless the LLM semantic authority and preserved
  table authority support it.

## Execution Checklist

1. Build a post-LLM loss audit directory under the campaign root.
2. Compare old child `11/13` and new child `113/115` by paper and by function
   unit.
3. For high-loss papers, inspect:
   - S2-4b raw response
   - S2-5 parsed semantic object
   - S2-7 function-unit activation and projected rows
   - Stage5 decision trace
4. Classify each loss as one of:
   - `llm_under_authorized`: raw response did not authorize the missing
     formulation/table/DOE scope.
   - `s2_5_parse_loss`: raw response contains semantics that S2-5 failed to
     preserve.
   - `s2_7_projection_loss`: S2-5 contains semantics, but S2-7 failed table or
     DOE reattachment/materialization.
   - `stage5_filter_loss`: rows reached Stage5 and were filtered/collapsed by
     a closure rule.
   - `old_lineage_over_expansion`: old rows lack lawful LLM semantic authority
     and should not be restored.
5. Match observed classes against `docs/repair_index/success_pattern_index_v1.tsv`
   and governed memory before changing code.
6. Apply only the smallest generic repair for the first confirmed deterministic
   loss class.
7. Run bounded replay from the earliest affected lawful boundary.
8. If bounded replay passes, run the aligned full campaign replay from the same
   lawful boundary through Stage5.
9. Update `CAMPAIGN_PROGRESS.md`, repair index, and memory with the result,
   including negative results.

## Initial Sentinel Papers

High-loss versus child `13`:

- `WFDTQ4VX`
- `UFXX9WXE`
- `7WLX2UBI`
- `5GIF3D8W`

High-gain / sanity controls:

- `QLYKLPKT`
- `GNQWKY3J`
- `ULCW6JTQ`

Zero-candidate S2-5 sample:

- Use the S2-5 summary from child `111` to select representative zero-candidate
  papers after the initial audit table is built.

## Acceptance Criteria

The plan is complete only when one of these outcomes is recorded:

- A generic deterministic post-LLM repair is implemented, bounded replay passes,
  and full campaign replay reaches Stage5 with an aligned comparison against
  child `115`; or
- The audited losses are shown to be `llm_under_authorized` or
  `old_lineage_over_expansion`, meaning no deterministic post-LLM repair is
  legal without changing prompt/model/live-call behavior.

## Execution Result

Status: `completed`

Implemented repair:

- `src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py`
  now allows an LLM-declared DOE scope to bind a source-text table anchor when
  the normalized table payload is missing or cannot provide a usable execution
  target.
- `src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py` now
  enumerates clean-text table blocks only inside the LLM-authorized DOE scope,
  prefers clean-text rows over lossy Marker/normalized payload rows when they
  expose more explicit table rows, and rejects source-text fallback when row
  labels are non-positive, duplicated, or non-contiguous.

Validation:

- Bounded replay recovered `UFXX9WXE` from unresolved/lossy DOE materialization
  to 26 explicit DOE rows and `7WLX2UBI` to 9 explicit table rows.
- The first full replay `117 -> 118 -> 119` exposed a regression on
  `6AT9RFVD`, where a coded factor-level table was incorrectly treated as
  formulation rows. The guard was tightened before acceptance.
- Guarded full replay `120 -> 121 -> 122` produced 1277 S2-7 rows and 1042
  Stage5 final rows, compared with 1250 S2-7 rows and 1011 Stage5 final rows
  in child `113 -> 114 -> 115`.
- Lineage-aligned post-repair audit found positive final-row deltas only for
  `UFXX9WXE` (+23) and `7WLX2UBI` (+8), and no negative paper-level final-row
  deltas.

Diagnostic artifacts:

- `data/results/20260511_b069802/116_post_llm_formulation_loss_audit/analysis/post_repair_guarded_lineage_aligned_summary_v1.json`
- `data/results/20260511_b069802/116_post_llm_formulation_loss_audit/analysis/post_repair_guarded_lineage_aligned_row_delta_v1.tsv`
- `data/results/20260511_b069802/120_stage2_s2_7_post_llm_doe_source_text_payload_repair_guarded`
- `data/results/20260511_b069802/121_stage3_post_llm_doe_source_text_payload_repair_guarded`
- `data/results/20260511_b069802/122_stage5_post_llm_doe_source_text_payload_repair_guarded`
