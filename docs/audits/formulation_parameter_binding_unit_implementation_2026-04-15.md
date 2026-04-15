# Formulation Parameter Binding Unit Implementation 2026-04-15

## 1. What changed

Created:

- `src/analysis/build_formulation_parameter_binding_unit_v1.py`
- `src/analysis/run_formulation_parameter_binding_unit_v1.py`

Updated:

- `src/stage5_benchmark/build_deterministic_step2_value_backfill_v1.py`
- `src/analysis/run_deterministic_step2_baseline_v1.py`

The new helper adds a deterministic formulation-level ownership surface for composition-side fields. It does not change Step 1 membership, does not fill values directly, and does not weaken the existing explicit-only Step 2 contract. Step 2 only consumes this new surface when `--parameter-binding-tsv` is explicitly provided.

## 2. Prior repo implementations reused as references

- `src/stage5_benchmark/build_deterministic_step2_value_backfill_v1.py`
- `src/stage3_relation/build_formulation_relation_artifacts_v1.py`
- `src/stage5_benchmark/build_minimal_final_output_v1.py`
- `src/analysis/build_table_row_binding_unit_v1.py`
- `src/analysis/run_table_row_binding_unit_v1.py`

The implemented v1 focuses on lawful reuse of `resolved_relation_fields_v1.tsv` plus frozen final-row identity surfaces such as:

- `representative_source_formulation_id`
- `source_candidate_ids`
- `source_candidate_labels`
- `relation_method_group_ids`

## 3. Exact binding rules implemented

The new binding helper writes:

- `formulation_parameter_binding_candidates_v1.tsv`
- `formulation_parameter_binding_resolved_v1.tsv`
- `formulation_parameter_binding_summary_v1.md`

Implemented deterministic rules:

1. `exact_source_candidate_relation_match`
   - if a resolved relation field exists for the frozen row’s representative/source candidate id
   - result status: `resolved_relation_context`

2. `article_native_label_relation_match`
   - if a resolved relation field does not match by candidate id, but its article-native label uniquely matches the frozen row’s source labels
   - result status: `resolved_article_native_match`

3. `method_group_uniform_value`
   - if the frozen row carries `relation_method_group_ids` and the target field has one uniform value across that method group
   - result status: `resolved_shared_context`

4. `paper_global_uniform_value`
   - if no narrower match exists and the paper-wide resolved-relation pool for the field contains one uniform value
   - result status: `resolved_shared_context`

5. `no_relation_surface_available`
   - used for `phase_ratio` and `drug_polymer_ratio` in this v1
   - result status: `unsupported_context`

Conservative rules:

- if multiple conflicting values survive a candidate/label/method-group check, the row is marked `ambiguous_multiple_targets`
- Step 2 upgrades only from this resolved binding surface; it does not guess values
- Step 2 emits these upgrades as `relation_carried_explicit`
- `drug_polymer_ratio` is still derived only after both `drug_feed_amount` and `polymer_amount` become safely supported on the same frozen row

## 4. Exact commands run

```powershell
python src/analysis/run_formulation_parameter_binding_unit_v1.py --step1-run-dir data/results/20260415_23c14f0/04_deterministic_step1_frozen_subset --execution-cue formulation_parameter_binding_subset_v1
python src/analysis/run_formulation_parameter_binding_unit_v1.py --step1-run-dir data/results/20260415_23c14f0/15_dev15_deterministic_two_step_diag_with_binding_v1 --paper-key UFXX9WXE --paper-key L3H2RS2H --paper-key BB3JUVW7 --paper-key V99GKZEI --execution-cue formulation_parameter_binding_composition_subset_v1
```

## 5. Papers tested

Frozen subset validation:

- `5ZXYABSU`
- `WIVUCMYG`

Broader composition-heavy diagnostic subset:

- `UFXX9WXE`
- `L3H2RS2H`
- `BB3JUVW7`
- `V99GKZEI`

## 6. Resolved formulation-parameter bindings by field

Frozen subset run:

- run path: `data/results/20260415_23c14f0/17_formulation_parameter_binding_subset_v1`
- resolved bindings total: `32`
- resolved by field:
  - `surfactant_name`: `32`
- ambiguous by field:
  - `surfactant_name`: `3`

Broader composition-heavy subset:

- run path: `data/results/20260415_23c14f0/17_formulation_parameter_binding_composition_subset_v1`
- resolved bindings total: `392`
- resolved by field:
  - `polymer_mw_kDa`: `48`
  - `la_ga_ratio`: `48`
  - `polymer_amount`: `66`
  - `drug_feed_amount`: `56`
  - `surfactant_concentration`: `48`
  - `surfactant_name`: `66`
  - `organic_solvent`: `60`
- ambiguous by field:
  - `drug_feed_amount`: `10`
  - `surfactant_concentration`: `12`

## 7. Step 2 improvement summary

Frozen subset:

- row count unchanged: `35 -> 35`
- `final_formulation_id` preserved exactly: `pass`
- composition support before -> after:
  - `polymer_mw_kDa`: `0 -> 0`
  - `la_ga_ratio`: `0 -> 0`
  - `polymer_amount`: `0 -> 0`
  - `drug_feed_amount`: `0 -> 0`
  - `surfactant_concentration`: `0 -> 0`
  - `phase_ratio`: `0 -> 0`
  - `drug_polymer_ratio`: `0 -> 0`

Broader composition-heavy subset:

- row count unchanged: `66 -> 66`
- `final_formulation_id` preserved exactly: `pass`
- support before -> after:
  - `polymer_mw_kDa`: `48 -> 48`
  - `la_ga_ratio`: `48 -> 48`
  - `polymer_amount`: `27 -> 66`
  - `drug_feed_amount`: `24 -> 56`
  - `surfactant_concentration`: `27 -> 53`
  - `phase_ratio`: `0 -> 0`
  - `drug_polymer_ratio`: `24 -> 56`

Interpretation:

- `polymer_mw_kDa` and `la_ga_ratio` did not increase because the current Step 2 helper was already lawfully relation-carrying them.
- The main gain came from composition-side fields that were present in `resolved_relation_fields_v1.tsv` but not yet wired into Step 2 ownership:
  - `plga_mass_mg -> polymer_amount`
  - `drug_feed_amount_text -> drug_feed_amount`
  - `surfactant_concentration_text -> surfactant_concentration`
- `drug_polymer_ratio` improved only as a downstream consequence of safer support on both inputs.

## 8. Effect on modeling-core rows

- frozen subset modeling-core rows: `0 -> 0`
- broader composition-heavy subset modeling-core rows: `17 -> 17`

The modeling-core count did not move because the core definition for this diagnostic slice is:

- `drug_name`
- `polymer_mw_kDa`
- `la_ga_ratio`
- `encapsulation_efficiency_percent`

This binding unit improves composition-side ownership substantially, but it does not create new `encapsulation_efficiency_percent` rows and it does not materially change `polymer_mw_kDa` or `la_ga_ratio` coverage in the tested subset.

## 9. Open limitations

- `phase_ratio` remains unresolved in this v1 because there is no lawful deterministic relation surface for it yet.
- `drug_polymer_ratio` still depends on explicit support for both inputs and is not bound directly.
- `surfactant_concentration` still has `12` ambiguous cases in the broader subset.
- `drug_feed_amount` still has `10` ambiguous cases in the broader subset.
- This validation intentionally stopped at a broader diagnostic subset rather than immediately rolling the new unit across full DEV15.

## 10. Recommended next target

Use this formulation-parameter binding unit as the new additive composition-side ownership layer downstream of frozen Step 1 runs and table-row binding. The next implementation target should be a controlled DEV15-wide diagnostic rerun that combines:

1. Step 1 deterministic baseline
2. table-row binding unit
3. formulation-parameter binding unit
4. Step 2 with both optional binding TSVs enabled

That rerun should quantify:

- DEV15-wide composition support deltas
- `drug_polymer_ratio` follow-on gains
- whether modeling-core rows increase once the combined measurement-side and composition-side binding layers are both active
