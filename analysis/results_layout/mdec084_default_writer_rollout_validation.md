# MDEC084 Default Writer Rollout Validation

## Scope

This note validates the default-writer rollout only.

Non-goals:

- no historical run migration
- no authority switch
- no scientific pipeline change
- no benchmark interpretation change

## Inputs Used For Validation

- code inspection of:
  - `src/utils/run_id.py`
  - `src/utils/run_preflight.py`
  - `src/stage2_sampling_labels/run_stage2_composite_v1.py`
  - `src/utils/run_threepaper_stage2_v2_comparison.py`
  - `src/stage5_benchmark/run_minimal_final_output_v1.py`
  - `src/stage3_relation/run_formulation_relation_artifacts_v1.py`
- registry and runbook sync checks:
  - `docs/maintained_script_surface.tsv`
  - `docs/src_script_registry.tsv`
  - `project/ACTIVE_PIPELINE_RUNBOOK.md`
  - `project/ACTIVE_PIPELINE_FLOW.md`
  - `project/FILE_NAMING_AND_VERSIONING.md`
- active authority check:
  - `data/results/ACTIVE_RUN.json`

## Validation Summary

### 1. Current active authority remains unchanged

- result: `pass`
- `data/results/ACTIVE_RUN.json` was not modified
- active run remains:
  - `run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1`

### 2. Historical runs remain readable

- result: `pass`
- legacy run parsing and explicit legacy `--run-id` compatibility remain in
  place
- active-data-source resolution for existing `run_*` surfaces was not removed

### 3. Maintained write entrypoints now default to v2 naming

- result: `pass`
- default-write rollout now applies to:
  - `src/stage2_sampling_labels/run_stage2_composite_v1.py`
  - `src/utils/run_threepaper_stage2_v2_comparison.py`
  - `src/stage5_benchmark/run_minimal_final_output_v1.py`
- `src/stage3_relation/run_formulation_relation_artifacts_v1.py` was also
  updated for consistency, even though it is a stage-local wrapper rather than
  a maintained mainline entrypoint

### 4. Explicit legacy mode still works intentionally

- result: `pass`
- maintained writers still accept explicit legacy `--run-id`
- default behavior no longer silently invents legacy `run_*` names

### 5. No targeted utility still creates legacy names by default

- result: `pass` for the maintained write surfaces changed in this rollout
- `src/utils/run_latest.py` remains legacy compatibility only, but it does not
  create `data/results` run roots and therefore is not a blocker for the v2
  default writer rollout

### 6. Contract consistency with FILE_NAMING_AND_VERSIONING

- result: `pass`
- current behavior now matches the future-facing rule already documented under
  MDEC084:
  - bucket: `YYYYMMDD_<short_hash>`
  - child: `NN_<cue>` or `NNN_<cue>`

## Validation Commands

```powershell
python -m py_compile src/utils/run_id.py src/utils/run_preflight.py src/stage2_sampling_labels/run_stage2_composite_v1.py src/utils/run_threepaper_stage2_v2_comparison.py src/stage5_benchmark/run_minimal_final_output_v1.py src/stage3_relation/run_formulation_relation_artifacts_v1.py
$env:PYTHONPATH='.'; python src/utils/run_preflight.py --subset subset5 --stage stage2
python src/stage2_sampling_labels/run_stage2_composite_v1.py --help
python src/utils/run_threepaper_stage2_v2_comparison.py --help
$env:PYTHONPATH='.'; python src/stage5_benchmark/run_minimal_final_output_v1.py --help
$env:PYTHONPATH='.'; python src/stage3_relation/run_formulation_relation_artifacts_v1.py --help
```

Observed default planning output from `run_preflight`:

- bucket: `data/results/20260331_5d9f4e6/`
- child: `data/results/20260331_5d9f4e6/01_stage2`

## Pilot Creation Decision

- tiny non-authoritative pilot run created: `no`

Reason:

- the default writer behavior was provable through shared helper resolution,
  CLI help, and code-path validation without adding another non-authoritative
  surface under `data/results/`
- skipping the pilot avoids unnecessary churn and any ambiguity around active
  authority or reviewer-facing artifacts

## Missing Pre-Read Inputs

These requested inputs were not present in the current workspace:

- `docs/mdec084_sync_note.md`
- `analysis/mdec084_real_writer_entrypoint_audit.md`
- `analysis/mdec084_real_writer_entrypoints.tsv`
- `analysis/mdec084_pilot_entrypoint_selection.md`

This rollout proceeded using the present MDEC084 contract, validation notes,
registry surfaces, and code inspection evidence only.
