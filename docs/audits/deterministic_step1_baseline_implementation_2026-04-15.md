# Deterministic Step 1 Baseline Implementation Report

## Executive Summary

- Implemented a manifest-driven deterministic Step 1 runner at `src/analysis/run_deterministic_step1_baseline_v1.py`.
- Kept the Step 1 chain aligned with the audit conclusion:
  1. `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`
  2. `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
  3. `src/stage3_relation/build_formulation_relation_artifacts_v1.py`
  4. `src/stage5_benchmark/build_minimal_final_output_v1.py`
  5. `src/stage5_benchmark/build_layer2_identity_scaffold_binding_v1.py`
  6. `src/stage5_benchmark/enforce_identity_freeze_v1.py`
- Preserved the no-LLM and no-external-API contract.
- Added failure-safe run logging so failed identity-freeze runs still write `RUN_CONTEXT.md` and command logs.

## Files Changed

- `src/analysis/run_deterministic_step1_baseline_v1.py`
- `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`

## What Changed

### 1. New Step 1 runner

`src/analysis/run_deterministic_step1_baseline_v1.py` now:

- accepts a manifest TSV derived from `data/cleaned/index/manifest_current.tsv`
- optionally filters by explicit `--paper-key`
- preflights deterministic eligibility from manifest rows without using LLM-backed artifacts
- writes:
  - `scoped_manifest.tsv`
  - `eligible_scope_manifest.tsv`
  - `eligible_corpus_inventory.tsv`
  - `requested_scope_keys.tsv`
  - `RUN_CONTEXT.md`
  - `command_execution_log_v1.json`
- runs the deterministic Step 1 chain end to end
- builds the Layer2 identity scaffold surface
- enforces identity freeze as a hard gate
- exits non-zero when identity freeze fails while still preserving run context

### 2. Emitter generalization support

`src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py` now exposes the governed supported-paper set through:

- `DOCUMENT_BUILDERS`
- `supported_paper_keys()`

This does not change extraction behavior. It lets the new runner resolve the lawful deterministic subset directly from a manifest instead of relying on a hardcoded external paper list.

## Why These Changes Were Needed

- The repo had the April 11 rules-only chain shape, but not a manifest-driven executable Step 1 surface with reproducibility-grade run context and explicit skip accounting.
- The emitter already supported multiple deterministic papers, but the supported set was not exposed as a reusable contract for runner-side eligibility resolution.
- The identity-freeze gate was already maintained, but there was no dedicated Step 1 runner wiring it in as the terminal success condition.

## Commands Run

```powershell
python src/analysis/run_deterministic_step1_baseline_v1.py --manifest-tsv data/cleaned/index/manifest_current.tsv --paper-key 5ZXYABSU --paper-key WIVUCMYG --execution-cue deterministic_step1_validation
python src/analysis/run_deterministic_step1_baseline_v1.py --manifest-tsv data/cleaned/index/manifest_current.tsv --execution-cue deterministic_step1_manifest_full_rerun
python src/analysis/run_deterministic_step1_baseline_v1.py --manifest-tsv data/cleaned/index/manifest_current.tsv --paper-key 5ZXYABSU --paper-key WIVUCMYG --execution-cue deterministic_step1_frozen_subset
```

## Run Outcomes

### Successful frozen Step 1 run

- Run path:
  - `data/results/20260415_23c14f0/04_deterministic_step1_frozen_subset`
- Manifest path:
  - `data/cleaned/index/manifest_current.tsv`
- Papers processed:
  - `2`
- Papers skipped:
  - `0`
- Processed paper keys:
  - `5ZXYABSU`
  - `WIVUCMYG`
- Final formulation row count:
  - `35`
- Identity freeze result:
  - `pass`

### Full manifest-driven deterministic attempt

- Run path:
  - `data/results/20260415_23c14f0/03_deterministic_step1_manifest_full_rerun`
- Manifest rows inspected:
  - `949`
- Eligible deterministic papers processed:
  - `15`
- Skipped rows:
  - `934`
- Skip reasons:
  - `931` = `no_text_path_in_manifest`
  - `3` = `unsupported_by_deterministic_emitter`
- Final formulation row count:
  - `199`
- Identity freeze result:
  - `fail`

## Identity Freeze Failure Detail For Full Scope

Identity freeze passed only for:

- `5ZXYABSU`
- `WIVUCMYG`

Identity freeze failed for:

- `5GIF3D8W`
- `7ZS858NS`
- `BB3JUVW7`
- `BXCV5XWB`
- `INMUTV7L`
- `L3H2RS2H`
- `PA3SPZ28`
- `QLYKLPKT`
- `RHMJWZX8`
- `UFXX9WXE`
- `V99GKZEI`
- `WFDTQ4VX`
- `YGA8VQKU`

Observed failure classes in `data/results/20260415_23c14f0/03_deterministic_step1_manifest_full_rerun/audit/identity_freeze_guardrail_v1/identity_freeze_summary_v1.tsv`:

- row-count drift
- identity reassignment
- unresolved scaffold rows

This means the deterministic Step 1 pipeline is now executable for manifest-driven scope, but it is not yet full-manifest-ready under the current freeze contract.

## Open Limitations Before Step 2

- Step 2 must not be implemented on top of the failing 15-paper full-scope run.
- The current passing frozen Step 1 scope is only the 2-paper subset:
  - `5ZXYABSU`
  - `WIVUCMYG`
- The remaining 13 deterministic papers need Step 1 identity-reconstruction repair before they can lawfully feed Step 2.
- The runner currently uses the repo-established scaffold baseline final table:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/44_stage5_descendant_fix_v1fixed_recompare/run_20260321_1454_5fa3ed0_dev15_stage5_descendant_fix_v1fixed_recompare_v1/final_formulation_table_v1.tsv`
- The run remains `diagnostic-only, not benchmark-valid final output` because no benchmark compare step was executed.

## Recommended Immediate Next Step

- Treat `data/results/20260415_23c14f0/04_deterministic_step1_frozen_subset` as the current successful deterministic Step 1 baseline artifact.
- Use `data/results/20260415_23c14f0/03_deterministic_step1_manifest_full_rerun` as the governed failure-localization surface for expanding Step 1 beyond the current 2-paper passing subset.
- Do not start Step 2 until the target Step 1 scope passes identity freeze.
