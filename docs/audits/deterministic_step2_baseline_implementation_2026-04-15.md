# Deterministic Step 2 Baseline Implementation Report

## Executive Summary

- Implemented the frozen-final explicit-only Step 2 builder at `src/stage5_benchmark/build_deterministic_step2_value_backfill_v1.py`.
- Added a thin explicit run-dir wrapper at `src/analysis/run_deterministic_step2_baseline_v1.py`.
- Validated Step 2 on the accepted 2-paper Step 1 subset:
  - source Step 1 run: `data/results/20260415_23c14f0/04_deterministic_step1_frozen_subset`
  - Step 2 run: `data/results/20260415_23c14f0/05_deterministic_step2_subset_v1`
- Row count matched Step 1 exactly and `final_formulation_id` preservation passed.

## Files Created Or Changed

- `src/stage5_benchmark/build_deterministic_step2_value_backfill_v1.py`
- `src/analysis/run_deterministic_step2_baseline_v1.py`

## Exact Commands Run

```powershell
python src/analysis/run_deterministic_step2_baseline_v1.py --step1-run-dir data/results/20260415_23c14f0/04_deterministic_step1_frozen_subset --execution-cue deterministic_step2_subset_v1
```

## Inputs

- Input Step 1 run path:
  - `data/results/20260415_23c14f0/04_deterministic_step1_frozen_subset`
- Source final table:
  - `data/results/20260415_23c14f0/04_deterministic_step1_frozen_subset/final_formulation_table_v1.tsv`
- Source relation records:
  - `data/results/20260415_23c14f0/04_deterministic_step1_frozen_subset/formulation_relation_v1/formulation_relation_records_v1.tsv`
- Source resolved relation fields:
  - `data/results/20260415_23c14f0/04_deterministic_step1_frozen_subset/formulation_relation_v1/resolved_relation_fields_v1.tsv`
- Source baseline identity assessment:
  - `data/results/20260415_23c14f0/04_deterministic_step1_frozen_subset/analysis/baseline_ready_identity_assessment_v1.tsv`

## Validation Outcome

- Processed paper count:
  - `2`
- Output row count:
  - `35`
- Row count matched Step 1 exactly:
  - `yes`
- ID preservation passed:
  - `yes`

## Output Artifacts

- `data/results/20260415_23c14f0/05_deterministic_step2_subset_v1/step2_value_backfill_table_v1.tsv`
- `data/results/20260415_23c14f0/05_deterministic_step2_subset_v1/step2_value_backfill_evidence_v1.tsv`
- `data/results/20260415_23c14f0/05_deterministic_step2_subset_v1/step2_value_backfill_summary_v1.md`
- `data/results/20260415_23c14f0/05_deterministic_step2_subset_v1/RUN_CONTEXT.md`
- `data/results/20260415_23c14f0/05_deterministic_step2_subset_v1/command_execution_log_v1.json`

## Explicit-Supported Fills By Field

Counts below treat both `explicit_supported` and `relation_carried_explicit` as successful explicit fills.

- `surfactant_name`: `32`
- `drug_name`: `6`
- all other maintained Step 2 fields: `0`

## Blank Counts By Field

Blank counts below count rows whose final Step 2 value cell remained empty, regardless of reason code.

- `polymer_mw_kDa`: `35`
- `la_ga_ratio`: `35`
- `surfactant_name`: `3`
- `surfactant_concentration`: `35`
- `organic_solvent`: `35`
- `drug_name`: `29`
- `drug_feed_amount`: `35`
- `polymer_amount`: `35`
- `drug_polymer_ratio`: `35`
- `phase_ratio`: `35`
- `encapsulation_efficiency_percent`: `35`
- `loading_capacity_percent`: `35`
- `particle_size_nm`: `35`
- `pdi`: `35`
- `zeta_potential_mV`: `35`

## Notes On Evidence Discipline

- `surfactant_name` was often filled through `relation_carried_explicit` from candidate-level resolved relation fields.
- `drug_name` was filled only where the frozen-final row already carried an explicit value.
- Table-like physicochemical outputs such as particle size, PDI, zeta potential, and encapsulation efficiency remained blank when the frozen-final surface exposed table-derived values without row-local table binding. Those rows were marked `unresolved_table` instead of being weakly filled.
- `drug_polymer_ratio` was left blank across the validated subset because the required explicit inputs were not jointly available in the same frozen formulation context.

## Open Limitations

- This Step 2 helper does not repair Step 1 and does not broaden Step 1 eligibility.
- It currently operates only on explicit frozen-final and relation-carried signals already available from Step 1 artifacts.
- It does not attempt direct fresh parsing from cleaned paper text or CSV table bodies beyond what Step 1 already exposed into frozen-final and relation artifacts.
- Broader Step 2 rollout still depends on Step 1 scope readiness. The helper was validated only on the currently accepted 2-paper frozen subset.
