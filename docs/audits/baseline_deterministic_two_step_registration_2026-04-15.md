# Baseline Deterministic Two-Step Registration (2026-04-15)

## Purpose

Register the deterministic no-LLM two-step DEV15 control baseline as a governed baseline object without changing pipeline logic, baseline schema columns, or any existing baseline entries.

## Registered baseline

- baseline id: `baseline_20260415_deterministic_two_step_v1`
- baseline type: `deterministic_two_step_baseline`
- benchmark validity: `diagnostic_only`
- intended use: `control_baseline_for_extraction_and_modeling_feasibility`

## What was added

- new manifest:
  - `data/baselines/baseline_20260415_deterministic_two_step_v1/BASELINE_MANIFEST.json`
- new registry row:
  - `data/baselines/BASELINE_REGISTRY.tsv`
- additive baseline-type vocabulary entry in:
  - `src/utils/baseline_registry_v1.py`

No existing baseline row or manifest was modified semantically.

## Classification decision

This baseline is explicitly classified as:

- deterministic
- no-LLM
- two-step
- binding-enhanced
- diagnostic-only control baseline

It is explicitly **not** classified as:

- replay baseline
- partial Stage2 baseline
- benchmark-valid baseline

## Authority and artifact note

The requested authority root for registration was:

- `data/results/20260415_23c14f0/18_dev15_deterministic_two_step_diag_with_combined_binding_v3`

That exact run directory is referenced in the governed implementation lineage and runner defaults, but it is not materialized in the current workspace copy. The registration therefore uses the documented authority root and the audited artifact-path contract for that lineage rather than a guessed replacement by recency.

Checked evidence used to support that registration choice:

- `docs/audits/deterministic_two_step_baseline_audit_2026-04-15.md`
- `docs/audits/deterministic_step2_baseline_implementation_2026-04-15.md`
- `docs/audits/table_row_binding_unit_implementation_2026-04-15.md`
- `docs/audits/formulation_parameter_binding_unit_implementation_2026-04-15.md`
- `src/analysis/run_dev15_deterministic_two_step_diagnostic_v1.py`

## Manifest content highlights

The new manifest records:

- Step1 deterministic identity reconstruction through Stage5 final table
- additive table-row binding and formulation-parameter binding surfaces
- explicit-only Step2 value backfill
- diagnostic compare outputs where stored
- required limitations for no-LLM, incomplete coverage, incomplete modeling-core overlap, and diagnostic-only status

## Validation commands

```powershell
python src/utils/baseline_registry_v1.py validate
python src/utils/baseline_registry_v1.py list
python src/utils/baseline_registry_v1.py show --query 20260415
```

## Expected interpretation

This new baseline should be used as the governed registration point for the strongest current deterministic no-LLM extraction control lineage for DEV15 under the explicit-only Step2 contract. It remains a diagnostic control surface, not a benchmark-valid final system baseline.
