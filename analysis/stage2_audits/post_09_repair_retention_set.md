# Post-09 Repair Retention Set

## Scope
- define the exact retained behavior set that must be active before a new integrated DEV15 baseline rerun
- include both code-level repairs and explicit replay-source choices
- keep the rerun within governed maintained entrypoints and bounded replay behavior

## Retained script-level behaviors
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
  - retain full-mode role-aware evidence preservation instead of unconditional `sorted_csv_first_4`
  - retain selected non-table authority blocks in `S2-2b`
  - retain sequential distinct-table supplementation from `S2-2a` candidate space
  - retain evidence-pack prompt construction when `selector_strategy=role_aware_selector_v1`
- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
  - retain governed Stage2 completion without changing semantic authority
  - retain routing into the existing DOE and sequential execution units
- `src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py`
  - retain lawful numbered DOE execution for `UFXX9WXE`-class papers when scope is present
- `src/stage2_sampling_labels/function_units/sequential_optimization_interpreter_v1.py`
  - retain governed sequential optimization resolution for `QLYKLPKT`

## Retained paper-proven capabilities
- `UFXX9WXE`
  - protect DOE authority-table preservation across `S2-2b`
  - preserve lawful activation of the existing DOE function unit
- `5GIF3D8W`
  - protect the repaired evidence handoff and the improved `8`-row replay floor
  - do not allow regression to the anchor’s `3`-row collapse
- `QLYKLPKT`
  - protect preservation of both true optimization tables at `S2-2b`
  - keep both table families present downstream even if row-count cleanup is still incomplete
- `WFDTQ4VX`
  - keep the anchor replay restoration that the `08...wfdtq4vx_restored` source introduced

## What the new baseline must preserve
- the current working-tree selector/prompt repair in `extract_semantic_stage2_objects_v2.py`
- explicit replay-source policy:
  - default replay directory: `data/results/20260417_385b6e1/08_dev15_stage2_baseline_wfdtq4vx_restored_v1/semantic_stage2_objects/raw_responses`
  - override `5GIF3D8W` with `data/results/20260417_385b6e1/06_dev15_full_baseline_no_marker_live/semantic_stage2_objects/raw_responses/5GIF3D8W__stage2_v2_raw_response.json`
  - override `QLYKLPKT` with `data/results/20260417_385b6e1/06_dev15_full_baseline_no_marker_live/semantic_stage2_objects/raw_responses/QLYKLPKT__stage2_v2_raw_response.json`
  - override `UFXX9WXE` with `data/results/20260417_385b6e1/06_dev15_full_baseline_no_marker_live/semantic_stage2_objects/raw_responses/UFXX9WXE__stage2_v2_raw_response.json`
- maintained entrypoints only:
  - `src/stage2_sampling_labels/run_stage2_composite_v1.py`
  - `src/stage3_relation/build_formulation_relation_artifacts_v1.py`
  - `src/stage5_benchmark/build_minimal_final_output_v1.py`
  - `src/stage5_benchmark/build_layer2_identity_scaffold_binding_v1.py`
  - `src/stage5_benchmark/enforce_identity_freeze_v1.py`
  - `src/stage5_benchmark/compare_final_table_to_gt_v1.py`

## Known partial capabilities entering the rerun
- `5GIF3D8W`
  - repaired expectation is partial: preserve the `8`-row improved floor, not full `24` or GT `26`
- `QLYKLPKT`
  - repaired expectation is partial: preserve both table families, but final output may still exceed the expected `7`
- `UFXX9WXE`
  - preserved expectation is stronger than the anchor, but success still depends on the retained best-available replay source and authority-table handoff

## Validation targets after rerun
- `UFXX9WXE`
  - `S2-2b` must preserve the authority table
  - Stage2/Stage5 must improve materially over the anchor’s `1` final row
- `5GIF3D8W`
  - integrated rerun must not fall back from `8` to `3`
- `QLYKLPKT`
  - integrated rerun must preserve both optimization-table families and stay above the anchor’s `3` final rows
- `WFDTQ4VX`
  - integrated rerun must not lose the anchor’s replay-restored behavior by accidentally reverting to the `06` replay file
