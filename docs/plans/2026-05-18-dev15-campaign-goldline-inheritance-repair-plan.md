# DEV15 Campaign Goldline Inheritance Repair Plan

created_at_utc: 2026-05-18T13:39:13Z

## Purpose

The campaign lineage under `data/results/20260511_b069802` must not be treated
as using the same DEV15-validated settings until the DEV15 15-paper subset
reproduces the active DEV15 final-row count.

Current observed failure:

- DEV15 active Stage5 final table: 202 rows across 15/15 papers.
- Campaign guarded Stage5 final table child `122`: 164 rows for the same
  DEV15 15 keys.
- Clean-text content hashes for representative regressed papers match between
  active and campaign.
- The first content-level divergence is S2-2 normalized table payload and
  evidence construction, not raw PDF/HTML text.

## Required Tasks

1. Locate why campaign S2-2 does not reproduce the active DEV15 normalized table
   payload content for the DEV15 15-paper subset.
2. Determine whether the cause is:
   - Stage1 manifest `table_dir` / `stage1_table_cell_sidecar` selection not
     equivalent to the DEV15 active goldline, or
   - the S2-2 table authority builder not consuming equivalent table assets in
     campaign mode.
3. Implement a general repair. Do not patch one paper at a time.
4. Replay only the DEV15 15-paper subset through the campaign lineage:
   - S2-2
   - S2-4a
   - same-parameter live or lawful replay boundary
   - S2-5
   - S2-6
   - S2-7
   - Stage3
   - Stage5
5. Acceptance target:
   - Campaign DEV15 subset final table must recover 202 rows across the same
     15 DEV15 keys before the large campaign row count is interpreted.

## Guardrails

- LLM semantic discovery remains the formulation-authority boundary.
- Deterministic repair may restore table authority, payload construction,
  locator binding, or replay legality only.
- No GT backfill.
- No deterministic semantic row creation outside LLM-authorized scope.
- No benchmark claim. This is diagnostic-only until a governed compare says
  otherwise.
- If raw live responses must be reused, only use a lawful replay boundary and
  record the exact source raw-response directory.

## Execution Log

- Completed 2026-05-18.

### Diagnosis

- Confirmed that representative DEV15 clean-text content matched between active
  and campaign, so the first material divergence was not PDF/HTML extraction.
- Found that Stage1 unified campaign replay was overwriting explicit
  `data/cleaned/goren_2025/tables/<key>` manifest table directories with
  Marker/sidecar-derived campaign-local table assets. This made campaign S2-2
  consume non-equivalent table authority for DEV15.
- After Stage1 repair, found a second S2-2 divergence: non-hard-drop table
  candidates were preserved in authority/debug surfaces but could be evicted
  from `prompt_selected_candidates` by prose prompt-health budgeting. Because
  normalized table payload construction consumes selected table evidence, this
  caused table payload loss.
- Found a replay-path remapping gap: S2-5 `resolve_tables_dir_for_record`
  did not normalize backslash table paths such as
  `data\cleaned\goren_2025\tables\5GIF3D8W`.
- Found one post-live final-row overcount in `BXCV5XWB`: live LLM emitted
  loaded-name aliases (`KGN-loaded PLGA... nanoparticles`) in addition to
  already materialized table rows (`PLGA`, `PLGA-PEG`, `PLGA-PEG-HA`).

### Repairs

- `src/stage1_cleaning/build_stage1_unified_current_marker_v1.py`
  preserves explicit manifest `table_dir` / `table_available` and only promotes
  Marker/sidecar table assets when no explicit table directory is present.
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
  now keeps a bounded structural table floor in S2-2 prompt selection and
  rebuilds inline-table payload items when replay/campaign selector candidates
  are persisted without in-memory `item.rows`.
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
  now remaps backslash `table_dir` values before S2-5 table-dir resolution.
- `src/stage5_benchmark/build_minimal_final_output_v1.py`
  now collapses paper-local loaded-name LLM aliases onto a unique existing
  table material-composition row when the material label key matches exactly.

### Replay Lineage

- Stage1 DEV15 bounded replay:
  `data/results/20260511_b069802/123_stage1_dev15_goldline_table_authority_replay`
- S2-2 no-live replay:
  `data/results/20260511_b069802/127_stage2_s2_2_dev15_goldline_optional_positive_table_floor_no_live`
- S2-4a prompt freeze:
  `data/results/20260511_b069802/128_stage2_s2_4a_dev15_goldline_optional_positive_table_floor_prompt_freeze`
- S2-4b live DeepSeek same-parameter calls:
  `data/results/20260511_b069802/129_stage2_s2_4b_dev15_goldline_optional_positive_table_floor_deepseek_live`
  plus timeout retry
  `data/results/20260511_b069802/130_stage2_s2_4b_dev15_5zxyabsu_retry_same_params_deepseek_live`
- Combined raw-response boundary:
  `data/results/20260511_b069802/131_stage2_s2_4b_dev15_combined_129_plus_130_raw_responses`
- S2-5:
  `data/results/20260511_b069802/133_stage2_s2_5_dev15_goldline_semantic_parse_path_remap`
- S2-6:
  `data/results/20260511_b069802/134_stage2_s2_6_dev15_goldline_contract_validation`
- S2-7:
  `data/results/20260511_b069802/135_stage2_s2_7_dev15_goldline_compatibility_projection`
- Stage3:
  `data/results/20260511_b069802/136_stage3_dev15_goldline_relation_artifacts`
- Stage5:
  `data/results/20260511_b069802/138_stage5_dev15_goldline_loaded_alias_duplicate_collapse`
- Diagnostic compare:
  `data/results/20260511_b069802/139_compare_dev15_goldline_loaded_alias_duplicate_collapse`

### Validation

- Unit tests:
  - `python3 -m unittest tests.test_stage1_unified_marker_table_promotion_v1 tests.test_stage2_composite_manifest_binding_refresh_v1 -q`
  - `python3 -m unittest tests.test_stage2_preparation_core_selector_floor_v1 -q`
  - `python3 -m py_compile src/stage5_benchmark/build_minimal_final_output_v1.py`
- Stage5 final table recovered 202 rows across the 15 DEV15 keys.
- Paper-level final-row counts match active 746 for all 15 papers.
- Diagnostic GT count compare run 139 reports:
  - papers_in_scope: 15
  - papers_matching: 15
  - papers_mismatching: 0
  - total_final_table_rows: 202
  - total_gt_rows: 202

This is a diagnostic replay result, not benchmark certification.
