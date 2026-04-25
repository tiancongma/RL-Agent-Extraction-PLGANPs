# Code Change Audit

## Baseline reference

- Last successful baseline run for this audit:
  - `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1`
- Why this baseline was selected:
  - its `RUN_CONTEXT.md` explicitly says it supersedes `06_dev15_full_baseline_no_marker_live`
  - it contains completed Stage2, Stage3, and Stage5 outputs
  - it is marked as the current managed DEV15 diagnostic baseline for debugging

## Current mainline

- Current lineage under audit:
  - `data/results/20260418_9538ec2`
- Current downstream comparison surface used here:
  - `data/results/20260418_9538ec2/28_diagnosis_baseline_restart_stepwise_v1`
- Current lawful completed Stage2 surface used here:
  - `data/results/20260418_9538ec2/32_diagnosis_restart_s2_7_v1/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`

## Changes

### `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`

- Type of change:
  - `selector change (S2-2)`
  - `prompt assembly change (S2-3)`
  - `schema/prompt payload change`

- Short description:
  - `render_full_table_block` now cleans table rows before rendering full-table excerpts, which changes the literal table text passed downstream.
  - New bounded role-aware table selection was added for full-table mode, replacing broad `sorted_csv_first_4` fallback for some papers with selector-driven table picks.
  - Sequential-optimization-specific supplementation was added to force at least two distinct optimization-supporting tables when the paper shows sequential signals.
  - A local optimization pack was added:
    - explicit carry-forward bridge text
    - preparation/setup bridge text
    - role-ranked optimization tables
  - Trimmed context fallback replaced raw-prefix fallback in the new path, with explicit dropping of front matter and assay-heavy segments.
  - `build_live_prompt` now uses evidence-pack layout whenever `selector_strategy == role_aware_selector_v1`, even when `STAGE2_INPUT_EVIDENCE_PACKING_MODE=off`.

- Key code locations observed:
  - table cleanup before prompt rendering: `extract_semantic_stage2_objects_v2.py:1971`
  - bounded role-aware table selection: `extract_semantic_stage2_objects_v2.py:2789`
  - local optimization pack and bridge extraction: `extract_semantic_stage2_objects_v2.py:3026`
  - trimmed context fallback: `extract_semantic_stage2_objects_v2.py:4295`
  - new evidence-block assembly path: `extract_semantic_stage2_objects_v2.py:4411`
  - prompt-layout switch to evidence-pack layout: `extract_semantic_stage2_objects_v2.py:5660`

- Impact hypothesis:
  - Positive:
    - enables targeted local evidence selection for sequential-optimization papers
    - reduces broad raw-prefix prompt pollution for some papers
    - improves DOE table targeting in some cases such as `UFXX9WXE`
  - Negative / risk:
    - selector promotion can still choose noisy or wrong tables if role scoring overvalues assay-bearing tables
    - the new local optimization logic is paper-class-specific and can improve one paper while harming others
    - prompt layout changed materially without reducing output schema burden, so prompt size and live-call fragility remain
    - mixed evidence packs now depend much more on selector correctness than the baseline fallback path did

### `src/stage2_sampling_labels/run_stage2_composite_v1.py`

- Type of change:
  - `runtime / source-binding support`

- Short description:
  - No new working-tree delta was observed in this audit relative to the current checkout.
  - The maintained file already contains authoritative Stage1 refresh helpers:
    - `refresh_stage2_text_bindings`
    - `refresh_stage2_table_bindings`
  - These helpers support the post-baseline repair that refreshes Stage2 text/table bindings from governed authority rather than stale manifest copies.

- Key code locations observed:
  - text binding refresh: `run_stage2_composite_v1.py:111`
  - table binding refresh: `run_stage2_composite_v1.py:146`

- Impact hypothesis:
  - This file is not the main driver of the current regression delta.
  - Its relevant impact is enabling lawful Stage1 asset reuse with refreshed bindings during the April 18 rebuild.
  - The dominant current behavior change is in `extract_semantic_stage2_objects_v2.py` plus the runtime shift to fresh live Gemini execution.

### Runtime / live-call mode delta (run-level, not a new code diff in this audit)

- Type of change:
  - `runtime / live-call change`

- Short description:
  - Baseline `09_dev15_stage2_baseline_repaired_contractfix_v1` used:
    - `source_mode: legacy_llm_replay`
    - `max_text_chars: 30000`
    - replayed raw responses from `08_dev15_stage2_baseline_wfdtq4vx_restored_v1`
  - Current April 18 rebuild used:
    - fresh `S2-2 -> S2-3 -> S2-4b` execution
    - `replay_mode: false`
    - `stage2_enable_numbered_doe_recovery: 1`
    - `stage2_doe_enumeration_mode: explicit_only`
    - fresh live Gemini calls, with only `9/15` initial successes at `03_s2_4`
    - only `10/15` papers reaching completed Stage2 at `32_diagnosis_restart_s2_7_v1`

- Impact hypothesis:
  - This is the largest non-code driver of artifact drift.
  - Four papers disappearing entirely in the current downstream artifacts is explained first by incomplete fresh live-call coverage, not only by selector quality.
  - Any comparison between baseline and current outputs must therefore distinguish:
    - selector/prompt behavior change
    - incomplete fresh runtime completion

## Short classification summary

- Biggest code delta:
  - `extract_semantic_stage2_objects_v2.py` promoted role-aware full-table selection into the default full-table path and changed prompt assembly to evidence-pack layout.
- Biggest runtime delta:
  - baseline replayed completed raw responses; current lineage re-ran fresh live calls and only partially completed.
- Most likely impact split:
  - `selector/prompt change` explains paper-specific quality shifts
  - `runtime/live-call incompleteness` explains total paper loss and large downstream count collapse

## Caveat

- This audit is diagnostic-only.
- Findings should be peer-reviewed before being used for major decisions.
