# Post-09 Repair Preservation Audit

## Scope
- baseline anchor: `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1`
- this audit is required because the anchor baseline regressed multiple already-proven capabilities (`UFXX9WXE`, `5GIF3D8W`, `QLYKLPKT`) even though the repo still contained bounded repair logic and stronger replay sources
- the goal before rerunning DEV15 is to separate:
  - code-level repairs that must remain active
  - replay-source choices that must be retained explicitly
  - diagnostic-only single-paper outputs that should not be promoted directly into a baseline

## Baseline anchor
- exact path: `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1`
- what it represents:
  - the managed DEV15 diagnostic baseline after the contract-fix pass
  - a full Stage2 -> Stage3 -> Stage5 lineage with diagnostic identity-freeze and compare outputs
  - a regression point for post-09 preservation work because it dropped or weakened:
    - `UFXX9WXE` DOE authority-table execution (`1` final row at the anchor)
    - `5GIF3D8W` restored formulation-universe coverage (`3` final rows at the anchor)
    - `QLYKLPKT` sequential two-table preservation (`3` final rows at the anchor)

## Post-baseline repair runs reviewed
- `data/results/20260417_385b6e1/10_5gif3d8w_capability_restoration_validation_v1`
- `data/results/20260417_385b6e1/11_5gif3d8w_capability_restoration_validation_v2`
- `data/results/20260417_385b6e1/11_qlyk_capability_restoration_bounded_v1`
- `data/results/20260417_385b6e1/12_qlyk_capability_restoration_replay_v1`
- `data/results/20260418_3579206/01_dev15_integrated_post09_repair_baseline_v1`
- `data/results/20260418_3579206/02_dev15_integrated_post09_repair_baseline_replay06_v1`
- `analysis/paper_repairs/qlyk/qlyk_table_loss_audit.md`
- `analysis/paper_repairs/qlyk/qlyk_table_loss_audit.tsv`
- `analysis/baseline_regressions/phase1_regression_guard_plan.md`
- `analysis/baseline_regressions/phase1_regression_guard_catalog.tsv`
- `analysis/baseline_regressions/historical_capability_regression_inventory.md`
- `analysis/baseline_regressions/historical_capability_regression_inventory.tsv`
- `project/4_DECISIONS_LOG.md`
- current working-tree patch:
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
- historical proof surfaces used for preservation targeting:
  - `data/results/20260417_385b6e1/06_dev15_full_baseline_no_marker_live`
  - `data/results/20260417_385b6e1/08_dev15_stage2_baseline_wfdtq4vx_restored_v1`
  - `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1`

## Script-level changes after baseline
- path: `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
  - what changed:
    - full-mode `S2-2b` no longer falls back blindly to `sorted_csv_first_4` when bounded role-aware table candidates exist
    - selected non-table authority blocks are now preserved in full mode
    - sequential-optimization papers can be supplemented to at least two distinct preserved table origins from already-segmented `S2-2a` candidates
    - `build_live_prompt` now consumes the curated evidence-pack layout whenever `selector_strategy=role_aware_selector_v1`, even with `input_packing_mode=off`
  - which capability it affects:
    - `5GIF3D8W` evidence-pack restoration from `3` to `8` final rows
    - `QLYKLPKT` two-table preservation at `S2-2b` and prompt coverage
    - `UFXX9WXE` authority-table preservation at `S2-2b` because the selector no longer has to drop high-priority table candidates in full mode
  - whether it must be preserved in the new baseline:
    - yes
- no committed post-09 Stage3 or Stage5 script change was found
- preserve-critical unchanged maintained dependencies:
  - `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
  - `src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py`
  - `src/stage2_sampling_labels/function_units/sequential_optimization_interpreter_v1.py`
  - these were not changed after the anchor, but omitting their existing governed behavior would reintroduce already-proven capability loss

## Paper-level repair summary
- paper key: `UFXX9WXE`
  - capability restored or partially restored:
    - historically proven governed DOE execution with `26` numbered DOE rows after valid DOE scope activation
    - preservation target at the post-09 boundary is narrower: keep the DOE authority table alive across `S2-2b` so the existing DOE execution unit can activate again
  - current status:
    - anchor baseline still regressed (`1` final row; feature report shows `numbered_doe_row_enumeration_priority=missing`)
    - no post-09 single-paper rerun was recorded under the `20260417_385b6e1` bucket
    - repo governance and historical validation still prove the capability exists and must not be dropped
  - whether this capability must be preserved in the new integrated baseline:
    - yes
- paper key: `5GIF3D8W`
  - capability restored or partially restored:
    - post-09 repaired handoff restores role-aware evidence selection, a non-empty execution-grade table payload, and a stronger replay source
    - final output improves from `3` to `8`
  - current status:
    - partially restored
    - optimized-table floor recovered
    - historical `24` sweep-row capability remains unrecovered because no stronger lawful replayable raw response is currently available in-repo
  - whether this capability must be preserved in the new integrated baseline:
    - yes
- paper key: `QLYKLPKT`
  - capability restored or partially restored:
    - post-09 repaired selector preserves both true optimization tables through `S2-2b`
    - prompt coverage now includes both tables
    - Stage2/Stage5 again materialize both table families
  - current status:
    - partially restored
    - row-span coverage returns, but replayed downstream output over-materializes (`24` Stage2 rows, `10` final rows instead of expected `7`)
  - whether this capability must be preserved in the new integrated baseline:
    - yes
- paper key: `WFDTQ4VX`
  - capability restored or partially restored:
    - pre-anchor anchor-critical replay restoration carried in `08_dev15_stage2_baseline_wfdtq4vx_restored_v1`
  - current status:
    - preserved in the anchor only because the anchor replay source switched away from `06`
    - not a post-09 repair, but a required anchor capability that would be lost if the new rerun used `06` blindly for every paper
  - whether this capability must be preserved in the new integrated baseline:
    - yes

## Preservation-critical changes
- preserve the current working-tree repair in `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
- preserve existing governed execution behavior in:
  - `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
  - `src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py`
  - `src/stage2_sampling_labels/function_units/sequential_optimization_interpreter_v1.py`
- preserve explicit replay-source choices instead of assuming one replay directory is globally best:
  - default anchor replay source: `data/results/20260417_385b6e1/08_dev15_stage2_baseline_wfdtq4vx_restored_v1/semantic_stage2_objects/raw_responses`
  - `5GIF3D8W` stronger replay source: `data/results/20260417_385b6e1/06_dev15_full_baseline_no_marker_live/semantic_stage2_objects/raw_responses/5GIF3D8W__stage2_v2_raw_response.json`
  - `QLYKLPKT` stronger replay source: `data/results/20260417_385b6e1/06_dev15_full_baseline_no_marker_live/semantic_stage2_objects/raw_responses/QLYKLPKT__stage2_v2_raw_response.json`
  - `UFXX9WXE` best available replay source on disk: `data/results/20260417_385b6e1/06_dev15_full_baseline_no_marker_live/semantic_stage2_objects/raw_responses/UFXX9WXE__stage2_v2_raw_response.json`
- preserve the post-09 interpretation that these replay choices are explicit run inputs, not implicit latest-code assumptions

## Non-promoted or diagnostic-only changes
- `10_5gif3d8w_capability_restoration_validation_v1` and `11_5gif3d8w_capability_restoration_validation_v2` are single-paper validation runs, not baseline candidates
- `11_qlyk_capability_restoration_bounded_v1` is `S2-4a` prompt-only and cannot stand in for an end-to-end repair
- `12_qlyk_capability_restoration_replay_v1` proves `S2-2b` restoration but remains over-materialized at `S2-5/S2-7`; its exact final output must not be promoted as the target behavior
- the historical `24`-row `5GIF3D8W` and `7`-row `QLYKLPKT` references are preservation anchors, not direct current-run outputs
- no fresh live `S2-4b` run is promoted here because the current environment provides no governed live-call path for this task

## Audit conclusion
- enough repaired capability exists to justify a new integrated DEV15 baseline:
  - yes, as a fully documented diagnostic rerun using the retained repair set and explicit replay-source curation
- main integration risks:
  - `UFXX9WXE`: the DOE function unit is already proven, but the new rerun will regress immediately if `S2-2b` drops the authority table again or if the best available replay source is not retained
  - the stronger DOE-scopefix raw-response path referenced in older decisions is not present under current `data/results/`, so UFXX preservation must be treated as partial unless the available `06` replay source is sufficient
  - `5GIF3D8W`: preserved improvement is only partial; a rerun can still drop back from `8` to `3` if it uses the anchor replay file instead of the stronger maintained replay source
  - `QLYKLPKT`: preserved table-handoff repair can still surface as `10` noisy final rows because the strongest lawful replay source remains over-materialized downstream
  - `WFDTQ4VX`: using `06` globally would silently discard the anchor’s pre-existing replay restoration
