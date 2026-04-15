# Core A Overlap Targeted Repair Round 2 (2026-04-15)

Diagnostic-only, not benchmark-valid final output.

## 1. Scope

This repair stayed intentionally narrow:

- target papers: `YGA8VQKU`, `BB3JUVW7`
- target fields: `polymer_mw_kDa`, `la_ga_ratio`
- target blocker class: `candidate_absent_upstream`
- membership contract: unchanged
- row identity contract: unchanged

Bounded execution surfaces:

- repaired bounded run: `data/results/20260415_targeted_core_a_repair_round2_codepath_v1/`
- repaired Step 2 output: `data/results/20260415_8a2502a/03_deterministic_step2_baseline/`
- prior comparison baseline: `data/results/20260415_targeted_core_a_repair_codepath_v2/`

## 2. Files Changed

- `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`
- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`

No formulation-membership logic, Stage 3 relation logic, or Step 2 contract logic was widened in this round.

## 3. Exact Logic Added

### 3.1 `YGA8VQKU` upstream polymer candidate recovery

Inside `build_yga8vqku_document`, the retained F-row DOE family now emits a paper-bounded polymer property:

- `molecular_weight = Low viscosity PLGA (0.32-0.44 dL/g)`

This is attached only to the retained F-row family and only as article-native text support.

### 3.2 `BB3JUVW7` upstream ratio candidate recovery

Inside `build_bb3juvw7_document`, the nanorod rows now emit:

- `la_ga_ratio = <PLGA type (lactide:glycolide)>`

This is taken directly from the explicit `PLGA type (lactide:glycolide)` column in the paper-local table.

### 3.3 Compatibility safeguard for viscosity descriptors

The compatibility projection now preserves viscosity-based polymer descriptors as text-only support and avoids misreading `dL/g` viscosity text as numeric `kDa`.

This prevents an unlawful numeric projection while still allowing the recovered upstream candidate to survive into the bounded deterministic path.

## 4. Why Each Change Was Justified By Checked Source Evidence

### 4.1 `YGA8VQKU`

Checked evidence:

- `data/cleaned/goren_2025/tables/YGA8VQKU/YGA8VQKU__table_01__html_table.csv`
- `data/cleaned/goren_2025/tables/YGA8VQKU/YGA8VQKU__table_07__html_table.csv`

Judgment:

- The retained F-row DOE family is the EE-positive family used in the bounded run.
- The paper-local summary table explicitly distinguishes low-viscosity and high-viscosity PLGA.
- The checked retained family aligns with the low-viscosity branch, so a family-level article-native polymer descriptor was lawful.
- The checked target surface still did not provide an equally explicit retained-family `la_ga_ratio`, so no ratio candidate was added for these EE-positive rows.

### 4.2 `BB3JUVW7`

Checked evidence:

- `data/cleaned/goren_2025/tables/BB3JUVW7/BB3JUVW7__table_02__html_table.csv`

Judgment:

- The table explicitly states `PLGA type (lactide:glycolide)` for the nanorod family.
- That explicit ratio signal was lawfully emitted upstream for those rows.
- The EE-positive rows in the bounded overlap target are the nanosphere rows, not the nanorod rows.
- The checked target surface still did not provide lawful `polymer_mw_kDa` or row-local `la_ga_ratio` for the EE-positive nanosphere rows, so no broader propagation was added.

## 5. Exact Commands Run

```powershell
python src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py --manifest-tsv data/cleaned/goren_2025/index/splits/dev_manifest_remaining12_2026-03-10.tsv --out-dir data/results/20260415_targeted_core_a_repair_round2_codepath_v1/semantic_stage2_objects --paper-keys YGA8VQKU BB3JUVW7
Import-Csv data/results/20260415_targeted_core_a_repair_codepath_v2/eligible_scope_manifest.tsv -Delimiter "`t" | Where-Object { $_.paper_key -in @('YGA8VQKU','BB3JUVW7') } | Export-Csv data/results/20260415_targeted_core_a_repair_round2_codepath_v1/eligible_scope_manifest.tsv -Delimiter "`t" -NoTypeInformation
python src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py --input-jsonl data/results/20260415_targeted_core_a_repair_round2_codepath_v1/semantic_stage2_objects/semantic_stage2_objects_v1.jsonl --output-dir data/results/20260415_targeted_core_a_repair_round2_codepath_v1/semantic_to_widerow_adapter
python src/stage3_relation/build_formulation_relation_artifacts_v1.py --weak-labels-tsv data/results/20260415_targeted_core_a_repair_round2_codepath_v1/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv --scope-manifest-tsv data/results/20260415_targeted_core_a_repair_round2_codepath_v1/eligible_scope_manifest.tsv --out-dir data/results/20260415_targeted_core_a_repair_round2_codepath_v1/formulation_relation_v1
python -m src.stage5_benchmark.build_minimal_final_output_v1 --input-tsv data/results/20260415_targeted_core_a_repair_round2_codepath_v1/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv --relation-records-tsv data/results/20260415_targeted_core_a_repair_round2_codepath_v1/formulation_relation_v1/formulation_relation_records_v1.tsv --resolved-relation-fields-tsv data/results/20260415_targeted_core_a_repair_round2_codepath_v1/formulation_relation_v1/resolved_relation_fields_v1.tsv --out-dir data/results/20260415_targeted_core_a_repair_round2_codepath_v1
python src/analysis/build_table_row_binding_unit_v1.py --final-table-tsv data/results/20260415_targeted_core_a_repair_round2_codepath_v1/final_formulation_table_v1.tsv --decision-trace-tsv data/results/20260415_targeted_core_a_repair_round2_codepath_v1/final_output_decision_trace_v1.tsv --relation-records-tsv data/results/20260415_targeted_core_a_repair_round2_codepath_v1/formulation_relation_v1/formulation_relation_records_v1.tsv --resolved-relation-fields-tsv data/results/20260415_targeted_core_a_repair_round2_codepath_v1/formulation_relation_v1/resolved_relation_fields_v1.tsv --scope-manifest-tsv data/results/20260415_targeted_core_a_repair_round2_codepath_v1/eligible_scope_manifest.tsv --out-dir data/results/20260415_targeted_core_a_repair_round2_codepath_v1
python src/analysis/build_formulation_parameter_binding_unit_v1.py --final-table-tsv data/results/20260415_targeted_core_a_repair_round2_codepath_v1/final_formulation_table_v1.tsv --decision-trace-tsv data/results/20260415_targeted_core_a_repair_round2_codepath_v1/final_output_decision_trace_v1.tsv --relation-records-tsv data/results/20260415_targeted_core_a_repair_round2_codepath_v1/formulation_relation_v1/formulation_relation_records_v1.tsv --resolved-relation-fields-tsv data/results/20260415_targeted_core_a_repair_round2_codepath_v1/formulation_relation_v1/resolved_relation_fields_v1.tsv --scope-manifest-tsv data/results/20260415_targeted_core_a_repair_round2_codepath_v1/eligible_scope_manifest.tsv --out-dir data/results/20260415_targeted_core_a_repair_round2_codepath_v1
python src/analysis/run_deterministic_step2_baseline_v1.py --step1-run-dir data/results/20260415_targeted_core_a_repair_round2_codepath_v1 --table-row-binding-tsv data/results/20260415_targeted_core_a_repair_round2_codepath_v1/table_row_binding_resolved_v1.tsv --parameter-binding-tsv data/results/20260415_targeted_core_a_repair_round2_codepath_v1/formulation_parameter_binding_resolved_v1.tsv
```

## 6. Before -> After Metrics For `YGA8VQKU` And `BB3JUVW7`

Combined:

- EE-positive rows: `21 -> 21`
- Core Set A rows: `0 -> 0`
- `polymer_mw_kDa` support: `0 -> 16`
- `la_ga_ratio` support: `0 -> 0`
- remaining `candidate_absent_upstream`: `21 -> 21`
- newly repaired into Core Set A: `0`

By paper:

| paper_key | EE-positive before | EE-positive after | Core A before | Core A after | polymer_mw_kDa before | polymer_mw_kDa after | la_ga_ratio before | la_ga_ratio after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `YGA8VQKU` | 16 | 16 | 0 | 0 | 0 | 16 | 0 | 0 |
| `BB3JUVW7` | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |

Detailed artifacts:

- `data/results/20260415_targeted_core_a_repair_round2_codepath_v1/analysis/target_paper_core_a_metrics_by_paper_v1.tsv`
- `data/results/20260415_targeted_core_a_repair_round2_codepath_v1/analysis/target_paper_core_a_row_status_v1.tsv`
- `data/results/20260415_targeted_core_a_repair_round2_codepath_v1/analysis/target_paper_core_a_summary_v1.tsv`

## 7. What Remains Unresolved

Remaining blocked EE-positive rows: `21`

- `YGA8VQKU` (`16` rows)
  - `polymer_mw_kDa` is now lawfully supported as article-native low-viscosity PLGA text.
  - `la_ga_ratio` remains absent on the checked EE-positive family path.
  - These rows therefore still fail Core Set A as `candidate_absent_upstream`.

- `BB3JUVW7` (`5` rows)
  - The recovered `la_ga_ratio` signal applies to the nanorod family, not to the EE-positive nanosphere rows.
  - The checked target surface still does not provide lawful `polymer_mw_kDa` for the EE-positive nanosphere rows.
  - These rows therefore still fail Core Set A as `candidate_absent_upstream`.

## 8. Is The Deterministic Baseline Still Worth Continuing After This Round?

Yes, but only with equally narrow evidence gates.

This round confirms that upstream candidate recovery can improve lawful support counts without changing formulation membership or `final_formulation_id`. It also shows the current checked target surfaces for `YGA8VQKU` and especially `BB3JUVW7` may not contain enough row-local explicit evidence to produce further Core Set A gains without crossing into speculative propagation.

The next extension is justified only if additional paper-local explicit polymer definition surfaces can be found for the EE-positive families. Without that, broader propagation would not be lawful under the current contract.
