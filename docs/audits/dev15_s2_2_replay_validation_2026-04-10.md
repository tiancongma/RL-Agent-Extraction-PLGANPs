# DEV15 S2-2 Replay Validation

Date: `2026-04-10`

Scope:
- maintained Stage2 only
- S2-2 boundary only: clean text -> governed evidence package
- no Stage3 benchmark work
- no Stage5 benchmark work
- no GT workbook scanning
- no new live LLM calls

Validation status:
- replay-based DEV15 full-set validation completed
- maintained Stage2 entrypoint used
- all 15 DEV15 papers completed

## Validation Setup

Maintained entrypoint used:
- `src/stage2_sampling_labels/run_stage2_composite_v1.py`

Explicit source manifest:
- `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv`

Explicit replay raw-response source:
- `data/results/20260402_5c1e7a4/01_dev15_raw_llm_freeze/frozen_raw_responses`

Why this replay source was used:
- it is a governed full-DEV15 raw freeze
- `RUN_CONTEXT.md` for that freeze records `raw outputs frozen: 15` and `missing: 0`
- it avoids new live LLM calls

Validation run output root:
- `data/results/20260410_a165cd1/02_dev15_s2_2_replay_validation`

Effective Stage2 input settings:
- `input_packing_mode = ordered_blocks`
- `table_mode = summary`
- `summary_first_column_enhancement = 1`
- ordered-block policy:
  - the run-level intended packing policy is `metadata > synthesis_method > materials_procurement > table > paragraph`
  - the exact resolved per-paper sequence is written into both:
    - `semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`
    - `analysis/stage2_prompt_preview_v1.tsv`

These settings are visible in:
- `RUN_CONTEXT.md`
- `evidence_blocks_v1.json` under `input_contract`
- `stage2_prompt_preview_v1.tsv`

## Run Command(s)

```powershell
$env:STAGE2_INPUT_EVIDENCE_PACKING_MODE='ordered_blocks'
$env:STAGE2_TABLE_MODE='summary'
$env:STAGE2_TABLE_SUMMARY_FIRST_COLUMN_ENHANCEMENT='1'
python src/stage2_sampling_labels/run_stage2_composite_v1.py `
  --run-dir data/results/20260410_a165cd1/02_dev15_s2_2_replay_validation `
  --manifest-tsv data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv `
  --source-mode legacy_llm_replay `
  --legacy-raw-responses-dir data/results/20260402_5c1e7a4/01_dev15_raw_llm_freeze/frozen_raw_responses `
  --llm-backend gemini `
  --model gemini-2.5-flash
```

Observed result:
- 15/15 papers completed
- `semantic_stage2_v2_objects.jsonl` written
- `semantic_stage2_v2_summary.tsv` written
- `semantic_stage2_objects/evidence_blocks/` written
- `analysis/stage2_prompt_preview_v1.tsv` written
- `analysis/table_selection_debug_v1.json` written
- `analysis/feature_activation_report_v1.tsv` refreshed
- `RUN_CONTEXT.md` refreshed

## Per-paper S2-2 Emission Summary

Summary outcome:
- evidence artifacts emitted: `15 / 15`
- prompt preview rows present: `15 / 15`
- summary-mode table debug payload present: `15 / 15`
- `technical_status.overall = pass`: `15 / 15`
- `design_status.overall = pass`: `15 / 15`
- preview references canonical artifact path: `15 / 15`

Per-paper details are recorded in:
- `docs/audits/dev15_s2_2_replay_validation_2026-04-10.tsv`

## Derivation Spot Checks

Representative paper 1:
- paper: `UFXX9WXE`
- reason: previously problematic paper and table-heavy case
- artifact:
  - `data/results/20260410_a165cd1/02_dev15_s2_2_replay_validation/semantic_stage2_objects/evidence_blocks/UFXX9WXE/evidence_blocks_v1.json`
- preview linkage:
  - `stage2_prompt_preview_v1.tsv` row for `UFXX9WXE` points to the exact artifact path
- contract consistency:
  - artifact order: `metadata > synthesis_method > materials_procurement > table > table > table > table > paragraph`
  - preview order: `metadata > synthesis_method > materials_procurement > table > table > table > table > paragraph`
  - artifact `technical_status.overall = pass`
  - preview `technical_status_overall = pass`
  - artifact `design_status.overall = pass`
  - preview `design_status_overall = pass`
- evidence profile:
  - `total_blocks = 8`
  - `table_blocks = 4`

Representative paper 2:
- paper: `5ZXYABSU`
- reason: simpler non-sequential spot check
- artifact:
  - `data/results/20260410_a165cd1/02_dev15_s2_2_replay_validation/semantic_stage2_objects/evidence_blocks/5ZXYABSU/evidence_blocks_v1.json`
- preview linkage:
  - preview row references the exact artifact path
- contract consistency:
  - artifact order: `metadata > synthesis_method > table > table > table > table > paragraph`
  - preview order: `metadata > synthesis_method > table > table > table > table > paragraph`
  - technical/design status match and both are `pass`
- evidence profile:
  - `total_blocks = 7`
  - `table_blocks = 4`

Representative paper 3:
- paper: `QLYKLPKT`
- reason: contract-sensitive sequential-optimization case
- artifact:
  - `data/results/20260410_a165cd1/02_dev15_s2_2_replay_validation/semantic_stage2_objects/evidence_blocks/QLYKLPKT/evidence_blocks_v1.json`
- preview linkage:
  - preview row references the exact artifact path
- contract consistency:
  - artifact order: `metadata > synthesis_method > table > table > table > table > paragraph`
  - preview order: `metadata > synthesis_method > table > table > table > table > paragraph`
  - technical/design status match and both are `pass`
- evidence profile:
  - `total_blocks = 7`
  - `table_blocks = 4`
  - `has_doe_signal = true`
  - `has_sequential_signal = true`

Strict derivation conclusion:
- for all 3 spot checks, prompt preview is explicitly tied to the canonical evidence artifact path
- for all 3 spot checks, block ordering and status fields are consistent between artifact and preview
- this supports the stronger run-wide result that `prompt_preview_rows=15 linked_rows=15`

## Governance Visibility Check

`RUN_CONTEXT.md`:
- records the formal S2-2 boundary
- records the canonical artifact path pattern
- records that prompt preview is derived observability
- records replay input settings:
  - `stage2_table_mode = summary`
  - `stage2_table_summary_first_column_enhancement = 1`
  - `stage2_input_evidence_packing_mode = ordered_blocks`

`analysis/feature_activation_report_v1.tsv` includes and marks active:
- `s2_2_evidence_artifact_contract`
  - `evidence_artifacts=15 valid_contract_artifacts=15`
- `s2_2_design_success_split`
  - `evidence_artifacts=15 with_status_split=15`
- `s2_2_prompt_preview_derived_from_evidence_artifact`
  - `prompt_preview_rows=15 linked_rows=15`

Also active for this run:
- `stage2_input_evidence_packing`

## Blockers / Anomalies

No S2-2 emission blockers were encountered in this validation run.

Observed anomalies:
- `RUN_CONTEXT.md` `run_activation_gate` is still `fail`, but not because of S2-2.
  - missing required features are:
    - `numbered_doe_row_enumeration_priority`
    - `variant_aware_gt_authority_switch`
  - these are outside the narrow S2-2 contract check requested here.
- `Boundary Governance` currently shows `replay_mode = llm_first_composite` while the run scope section correctly shows `source_mode = legacy_llm_replay`.
  - the actual replay inputs are correctly recorded in `source_files` and `legacy_raw_responses_dir`
  - this is a metadata-label mismatch, not an S2-2 artifact emission failure

Residual gap:
- none for DEV15 replay coverage in this run
- full-set validation was feasible without live LLM calls because the governed raw freeze covered all 15 DEV15 papers

## Recommendation For Manual Article Inspection

Recommended papers:
- `UFXX9WXE`
  - previously problematic paper
  - table-heavy
  - useful for checking the richest resolved block order and the canonical artifact-to-preview linkage
- `QLYKLPKT`
  - representative contract-sensitive paper with sequential and DOE-like signals
  - useful for checking whether the artifact snapshot and preview remain aligned on a more nuanced case
- `5ZXYABSU`
  - simpler representative case
  - useful for confirming the maintained contract also behaves cleanly on a less complicated paper
