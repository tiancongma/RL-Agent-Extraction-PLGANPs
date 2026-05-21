# EE35 Post-LLM Loss Localization And Repair Plan

## Status
`completed`

## Scope
This plan targets the 35-paper EE diagnostic subset under campaign
`data/results/20260511_b069802`.

Primary question:

> For the large-deficit papers, did S2-4b raw responses identify many DOE or
> formulation rows, and were they lost later in S2-5/S2-7; or did the raw
> response itself under-identify the formulation universe?

This is diagnostic-only. It must not be reported as benchmark-valid final
output.

## Governing Inputs
- selected keys:
  - `data/results/20260511_b069802/15_ee35_gt_selection/ee35_selected_keys_v1.txt`
- GPT Web master diagnostic reference:
  - `data/results/20260511_b069802/17_ee35_master_formulation_gt_from_gpt_web/final_table/ee35_master_formulation_gt_final_table_draft_v3.tsv`
- current repaired S2-4b raw response boundary:
  - `data/results/20260511_b069802/145_stage2_s2_4b_ee35_combined_replay108_live143_retry144_raw_responses/raw_responses`
- current S2-5 semantic parse:
  - `data/results/20260511_b069802/146_stage2_s2_5_ee35_goldline_semantic_parse/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
- current S2-7 compatibility projection:
  - `data/results/20260511_b069802/148_stage2_s2_7_ee35_goldline_compatibility_projection/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- current Stage5 final diagnostic output:
  - `data/results/20260511_b069802/150_stage5_ee35_goldline_final_output/final_formulation_table_v1.tsv`

## Relevant Repair Patterns To Check
- `PAT_S2_7_SOURCE_TEXT_DOE_PAYLOAD_FALLBACK_GUARDED_V1`
- `PAT_S2_7_TABLE_IDENTITY_ALIAS_LOCATOR_PRIORITY_V1`
- `PAT_S2_2_MULTI_EVIDENCE_ANCHOR_PRESERVATION_V1`
- `PAT_S2_2_METHOD_RESULT_UNDERSELECTION_FLOOR_V1`
- `PAT_UFXX9WXE_DOE_MAINLINE_RESTORE_V1`
- `PAT_TABLE_SCOPE_ID_SYNTHESIS_V1`

## Execution Plan

1. Build a paper-level loss localization audit.
   - Count GPT master rows per paper.
   - Count raw S2-4b formulation candidates, table scopes, DOE declarations,
     table formulation scopes, and any row enumeration fields per paper.
   - Count S2-5 parsed formulation candidates per paper.
   - Count S2-7 projected rows and Stage5 final rows per paper.
   - Classify loss boundary as:
     - `raw_underselected`
     - `s2_5_parse_loss`
     - `s2_7_projection_loss`
     - `stage5_filter_loss`
     - `mixed_or_unknown`

2. Deep-audit the largest-deficit papers.
   - Prioritize:
     - `JRMKHP5C`
     - `2RNHC2M5`
     - `XDIRIJ74`
     - `TIDBBF25`
     - `RM4BRF9X`
     - `PNKAM3D7`
     - `DCSRFP8X`
   - Inspect raw response JSON content, S2-5 object content, S2-7 row output,
     and current prompt/evidence surfaces.

3. If raw S2-4b already contains many row-level discoveries, repair the
   deterministic S2-5/S2-7 parser/projection/DOE expansion function that drops
   them.
   - Reuse existing repair patterns before adding new behavior.
   - Do not create rows without LLM semantic authorization.

4. If raw S2-4b itself under-identifies rows, do not patch Stage5 or invent
   semantic rows downstream.
   - Localize whether the raw under-selection correlates with prompt evidence,
     table summary truncation, DOE prompt overload, or missing table authority.
   - Record the next required repair boundary as S2-2/S2-4a/S2-4b rather than
     misattributing it to S2-7 or Stage5.

5. Run bounded replay after any code repair.
   - At minimum replay the affected high-deficit paper set through:
     `S2-5 -> S2-6 -> S2-7 -> Stage3 -> Stage5`.
   - If the repair is pre-LLM only, replay only through the legal no-live gate
     unless a fresh live call is explicitly required.

6. Validate and update this plan.
   - Write audit artifacts under a new campaign child run.
   - Record final classification and whether a code repair was applied.
   - Do not stop after only one paper-specific finding; report the aggregate
     boundary classification for all 35 papers and specific findings for the
     largest-deficit papers.

## Acceptance Criteria
- A repository-local audit artifact exists with raw/S2-5/S2-7/Stage5 counts
  for all 35 papers.
- The largest-deficit papers are classified by loss boundary.
- If deterministic post-LLM loss exists and a general repair is possible, the
  repair is implemented and bounded replay is run.
- If the dominant loss is raw S2-4b under-selection, the result is recorded as
  such, with evidence that downstream deterministic replay cannot lawfully
  recover missing semantic rows without another S2-2/S2-4a/S2-4b repair.

## Execution Result

Completed on 2026-05-18.

### Repair Applied
- Repaired `src/stage2_sampling_labels/run_stage2_s2_5_semantic_parsing_v1.py`
  so lawful combined raw-response replay boundaries can explicitly reattach the
  S2-2 authority run and `normalized_table_payloads` root.
- Repaired `src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py`
  so `t001` and `Table 1` aliases that resolve to the same normalized payload
  do not double-expand the same DOE table.
- Added regression coverage in
  `tests/test_stage2_s2_5_authority_reattachment_v1.py` and
  `tests/test_stage2_table_row_expansion_scope_alias_roles_v1.py`.
- Registered the repair pattern in
  `docs/repair_index/success_pattern_index_v1.tsv` as
  `PAT_S2_5_COMBINED_RAW_AUTHORITY_REATTACHMENT_OVERRIDE_V1`.

### Replay Lineage
- S2-5 repaired parse:
  `data/results/20260511_b069802/151_stage2_s2_5_ee35_authority_reattached_semantic_parse`
- S2-6 validation:
  `data/results/20260511_b069802/152_stage2_s2_6_ee35_authority_reattached_validation`
- S2-7 projection:
  `data/results/20260511_b069802/157_stage2_s2_7_ee35_authority_reattached_dedup_projection`
- Stage3 relation artifacts:
  `data/results/20260511_b069802/158_stage3_ee35_authority_reattached_dedup_relation_artifacts`
- Stage5 final diagnostic output:
  `data/results/20260511_b069802/159_stage5_ee35_authority_reattached_dedup_final_output`
- Loss localization audit:
  `data/results/20260511_b069802/160_ee35_post_llm_loss_localization_audit_dedup`

### Results
- S2-5 authority reattachment sidecar changed from unresolved to resolved for
  `35/35` papers using explicit CLI authority override.
- S2-7 rows improved from `107` to `145` after DOE alias target deduplication.
- Stage5 final rows improved from `94` to `107`.
- `2RNHC2M5` recovered from `4` S2-7 rows to `16` S2-7 rows after S2-2
  table payload reattachment and alias deduplication; the DOE unit emitted
  `14` rows.
- Remaining dominant deficit is raw S2-4b under-selection:
  the largest-deficit papers such as `JRMKHP5C`, `XDIRIJ74`, `TIDBBF25`,
  `RM4BRF9X`, `IC5L6Z3X`, and `ZB76MB3J` already have only `1-3` raw S2-4b
  formulation candidates, so S2-5/S2-7/Stage5 cannot lawfully recover their
  missing formulation universe without another pre-LLM/prompt/live-call repair.

### Audit Artifacts
- Paper-level localization:
  `data/results/20260511_b069802/160_ee35_post_llm_loss_localization_audit_dedup/analysis/ee35_post_llm_loss_localization_v2.tsv`
- Summary:
  `data/results/20260511_b069802/160_ee35_post_llm_loss_localization_audit_dedup/analysis/ee35_post_llm_loss_localization_summary_v2.md`
