# Table Row Binding Unit Implementation

Date: 2026-04-15

## 1. What changed

Files created:

- `src/analysis/build_table_row_binding_unit_v1.py`
- `src/analysis/run_table_row_binding_unit_v1.py`

Files changed:

- `src/stage5_benchmark/build_deterministic_step2_value_backfill_v1.py`
- `src/analysis/run_deterministic_step2_baseline_v1.py`

Governance note:

- The original target name under `src/stage5_benchmark/` was rejected by repo governance for new file creation in an active stage namespace.
- The new binding helper was therefore placed under `src/analysis/` and wired into Step 2 additively through `--table-row-binding-tsv`.

## 2. Prior repo implementations reused as references

Primary references inspected and reused:

- `src/stage2_sampling_labels/table_row_expansion_v1.py`
  - reused the idea of stable `table_id::row_XX` row keys and conservative row-level table identity handling
- `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`
  - reused paper-specific evidence patterns where supported rows already carry table-row snippets and article-native row labels
- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
  - reused the current measurement-field naming and the understanding that numeric fields are projected from `measurement_candidate`
- `src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py`
  - reused the idea that explicit row anchors and numbered rows are lawful deterministic recovery points
- `src/stage5_benchmark/build_deterministic_step2_value_backfill_v1.py`
  - preserved the explicit-only support contract and added only an optional row-binding upgrade path
- `src/stage5_benchmark/build_field_gt_review_workbook_v1.py`
  - reused the distinction between `supported` and `unresolved_table`
- `src/stage5_benchmark/export_final_formulation_audit_ready_v1.py`
  - reused frozen-final provenance directionality and row-level evidence handling expectations

## 3. Exact binding rules implemented

The new binding helper resolves row-local ownership only when one of these deterministic rules succeeds:

1. `supporting_snippet_exact`
   - the frozen Step 1 row already carries a table-row snippet in `supporting_evidence_refs`
   - exactly one table row matches that snippet

2. `article_label_exact`
   - the frozen Step 1 row preserves an article-native row/formulation label
   - exactly one table row matches that label

3. `snippet_and_label_overlap`
   - snippet matching and label matching both point to the same single row

4. `unique_numeric_signature`
   - no exact row label or snippet is enough by itself
   - exactly one row matches a multi-field numeric signature from the frozen final row
   - minimum signature strength used in v1: at least 2 matching numeric fields

5. `combined_unique_candidate`
   - the union of snippet/label/signature candidates collapses to one unique row

The helper does not:

- create new formulation rows
- alter final membership
- fill values directly
- guess between multiple candidate rows

The Step 2 upgrade rule added is:

- if a field is currently `unresolved_table`
- and a `resolved_row_local` binding exists for `(final_formulation_id, field_name)`
- Step 2 upgrades that field to `explicit_supported`
- otherwise existing Step 2 behavior is unchanged

## 4. Commands run

Successful validation commands:

```powershell
python src/analysis/run_table_row_binding_unit_v1.py --step1-run-dir data/results/20260415_23c14f0/04_deterministic_step1_frozen_subset --execution-cue table_row_binding_subset_v2
python src/analysis/run_table_row_binding_unit_v1.py --step1-run-dir data/results/20260415_23c14f0/08_dev15_deterministic_two_step_diag_v2 --paper-key WIVUCMYG --paper-key YGA8VQKU --paper-key V99GKZEI --execution-cue table_row_binding_numeric_subset_v3
```

Validated run directories:

- subset validation:
  - `data/results/20260415_23c14f0/12_table_row_binding_subset_v2`
- broader numeric subset:
  - `data/results/20260415_23c14f0/14_table_row_binding_numeric_subset_v3`

## 5. Papers tested

Bounded passing subset:

- `5ZXYABSU`
- `WIVUCMYG`

Broader diagnostic numeric subset:

- `WIVUCMYG`
- `YGA8VQKU`
- `V99GKZEI`

## 6. Validation contract checks

Subset run:

- Step 2 before row count: `35`
- Step 2 after row count: `35`
- `final_formulation_id` preserved exactly: `yes`

Broader numeric subset:

- Step 2 before row count: `48`
- Step 2 after row count: `48`
- `final_formulation_id` preserved exactly: `yes`

## 7. Resolved row-local bindings by field

### 7.1 Passing subset run

Run: `12_table_row_binding_subset_v2`

- total resolved row-local bindings: `104`
- `encapsulation_efficiency_percent`: `26`
- `particle_size_nm`: `26`
- `pdi`: `26`
- `zeta_potential_mV`: `26`
- `loading_capacity_percent`: `0`
- `polymer_amount`: `0`
- `drug_feed_amount`: `0`
- `surfactant_concentration`: `0`

### 7.2 Broader numeric subset run

Run: `14_table_row_binding_numeric_subset_v3`

- total resolved row-local bindings: `176`
- `encapsulation_efficiency_percent`: `48`
- `particle_size_nm`: `48`
- `pdi`: `32`
- `zeta_potential_mV`: `42`
- `loading_capacity_percent`: `6`
- `polymer_amount`: `0`
- `drug_feed_amount`: `0`
- `surfactant_concentration`: `0`

## 8. Ambiguous cases by field

Subset run:

- all prioritized fields: `0`

Broader numeric subset run:

- all prioritized fields: `0`

No row-local binding was accepted from an ambiguous multi-row candidate pool in either validation run.

## 9. Step 2 improvement summary

### 9.1 Passing subset run

Run: `12_table_row_binding_subset_v2`

- `encapsulation_efficiency_percent` explicit_supported: `0 -> 26`
- `particle_size_nm` explicit_supported: `0 -> 26`
- `pdi` explicit_supported: `0 -> 26`
- `zeta_potential_mV` explicit_supported: `0 -> 26`
- `loading_capacity_percent` explicit_supported: `0 -> 0`

### 9.2 Broader numeric subset run

Run: `14_table_row_binding_numeric_subset_v3`

- `encapsulation_efficiency_percent` explicit_supported: `0 -> 48`
- `particle_size_nm` explicit_supported: `0 -> 48`
- `pdi` explicit_supported: `0 -> 32`
- `zeta_potential_mV` explicit_supported: `0 -> 42`
- `loading_capacity_percent` explicit_supported: `0 -> 6`

## 10. Open limitations

1. v1 is strongest on papers where the frozen Step 1 row already preserves:
   - article-native table row labels such as `F14`
   - or exact row snippets in `supporting_evidence_refs`

2. v1 does not yet bind:
   - polymer amount
   - drug feed amount
   - surfactant concentration
   for the validated subsets, because those fields need additional table-header aliasing or stronger composition-side row matching on the tested papers.

3. v1 focuses on narrow numeric table-binding recovery and does not attempt a generic full-table ontology.

4. The broader numeric subset used an existing diagnostic Step 1 run, not a newly repaired full DEV15 legality surface.

## 11. Exact next implementation target

The next best extension is:

- broaden deterministic header/column alias coverage and composition-side row matching for:
  - `polymer_amount`
  - `drug_feed_amount`
  - `surfactant_concentration`
- while keeping the same conservative acceptance rule:
  no upgrade unless row ownership is uniquely determined

