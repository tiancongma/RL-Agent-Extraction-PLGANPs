# Core A Overlap Targeted Repair (2026-04-15)

Diagnostic-only, not benchmark-valid final output.

## 1. Scope

This repair stayed intentionally narrow:

- target papers: `WIVUCMYG`, `YGA8VQKU`, `V99GKZEI`, `BB3JUVW7`
- target fields: `drug_name`, `polymer_mw_kDa`, `la_ga_ratio`
- target blocker classes: `candidate_absent_upstream`, `ownership_unresolved`
- membership contract: unchanged
- row identity contract: unchanged

The bounded execution run is recorded at:

- `data/results/20260415_targeted_core_a_repair_codepath_v2/`
- Step 2 output: `data/results/20260415_8a2502a/02_deterministic_step2_baseline/`

## 2. Files Changed

- `src/stage3_relation/build_formulation_relation_artifacts_v1.py`
- `src/analysis/build_formulation_parameter_binding_unit_v1.py`
- `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`
- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`

Only the Stage 3 and formulation-parameter-binding changes were exercised in the final bounded code-path rerun. The Stage 2 emitter / compatibility edits were prepared as the matching upstream repair direction, but the bounded validation run seeded from the governed frozen weak-label surface so the repair could be validated without the local optional-dependency chain used by the compatibility stack.

## 3. Exact Logic Added

### 3.1 Ownership repair

- Added `drug_name` to `RESOLVABLE_RELATION_FIELDS` in `src/stage3_relation/build_formulation_relation_artifacts_v1.py`.
- Added `FieldSpec("drug_name", ("drug_name",), "tier1")` to `src/analysis/build_formulation_parameter_binding_unit_v1.py`.

Effect:

- `drug_name` now upgrades through the existing relation-backed ownership path instead of remaining stuck as `unresolved_table`.
- No donor fill or row membership change was introduced.

### 3.2 Target-scoped polymer recovery

Added paper-bounded Stage 3 supplements for the two papers where the repo-local evidence was strong enough:

- `WIVUCMYG`
  - carry `la_ga_ratio = 75:25`
  - carry `polymer_mw_kDa = Resomer 753S (PLGA grade)`
  - evidence anchor: `data/cleaned/content/text/WIVUCMYG.html.txt`
- `V99GKZEI`
  - carry `la_ga_ratio = 50:50`
  - carry `polymer_mw_kDa = RG502H MW range 7000-17000 Da`
  - evidence anchor: `data/cleaned/labels/manual/dev15_formulation_skeleton/candidates/V99GKZEI__10_1039_c5ra27386b__candidates.jsonl`

These supplements were inserted into the same `candidate_field_membership` relation family consumed by the maintained Stage 3 resolution logic. No broad cross-paper propagation was added.

## 4. Why The Changes Were Justified

The overlap-failure audit showed:

- all 53 target-scope EE-positive failures were blocked on `drug_name`, `polymer_mw_kDa`, and `la_ga_ratio`
- `drug_name` failed as `ownership_unresolved`
- `polymer_mw_kDa` and `la_ga_ratio` failed as `candidate_absent_upstream`

The paper-local evidence supported the narrow recovery choice:

- `WIVUCMYG` explicitly names `PLGA Resomer® 753S` in the cleaned paper text.
- `V99GKZEI` candidate notes explicitly state `PLGA (RG502H) MW range was 7000-17000 Da`.
- Repo-local historical guidance already treats `Resomer`/`RG` grades as lawful `polymer_mw_kDa.value_text` support when only the product grade is explicit.

The remaining two papers were left conservative:

- `YGA8VQKU` retained the known low/high-viscosity signal, but this bounded repair did not elevate it into `Core Set A` because the explicit LA:GA support for the EE-positive rows was still not established from the checked target surface.
- `BB3JUVW7` retained lawful `drug_name` ownership, but the EE-positive nanosphere rows still lacked explicit `polymer_mw_kDa` and row-local `la_ga_ratio` support.

## 5. Exact Commands Run

```powershell
python src/stage3_relation/build_formulation_relation_artifacts_v1.py --weak-labels-tsv data/results/20260415_targeted_core_a_repair_codepath_v2/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv --scope-manifest-tsv data/results/20260415_targeted_core_a_repair_codepath_v2/eligible_scope_manifest.tsv --out-dir data/results/20260415_targeted_core_a_repair_codepath_v2/formulation_relation_v1
python -m src.stage5_benchmark.build_minimal_final_output_v1 --input-tsv data/results/20260415_targeted_core_a_repair_codepath_v2/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv --relation-records-tsv data/results/20260415_targeted_core_a_repair_codepath_v2/formulation_relation_v1/formulation_relation_records_v1.tsv --resolved-relation-fields-tsv data/results/20260415_targeted_core_a_repair_codepath_v2/formulation_relation_v1/resolved_relation_fields_v1.tsv --out-dir data/results/20260415_targeted_core_a_repair_codepath_v2
python src/analysis/build_table_row_binding_unit_v1.py --final-table-tsv data/results/20260415_targeted_core_a_repair_codepath_v2/final_formulation_table_v1.tsv --decision-trace-tsv data/results/20260415_targeted_core_a_repair_codepath_v2/final_output_decision_trace_v1.tsv --relation-records-tsv data/results/20260415_targeted_core_a_repair_codepath_v2/formulation_relation_v1/formulation_relation_records_v1.tsv --resolved-relation-fields-tsv data/results/20260415_targeted_core_a_repair_codepath_v2/formulation_relation_v1/resolved_relation_fields_v1.tsv --scope-manifest-tsv data/results/20260415_targeted_core_a_repair_codepath_v2/eligible_scope_manifest.tsv --out-dir data/results/20260415_targeted_core_a_repair_codepath_v2
python src/analysis/build_formulation_parameter_binding_unit_v1.py --final-table-tsv data/results/20260415_targeted_core_a_repair_codepath_v2/final_formulation_table_v1.tsv --decision-trace-tsv data/results/20260415_targeted_core_a_repair_codepath_v2/final_output_decision_trace_v1.tsv --relation-records-tsv data/results/20260415_targeted_core_a_repair_codepath_v2/formulation_relation_v1/formulation_relation_records_v1.tsv --resolved-relation-fields-tsv data/results/20260415_targeted_core_a_repair_codepath_v2/formulation_relation_v1/resolved_relation_fields_v1.tsv --scope-manifest-tsv data/results/20260415_targeted_core_a_repair_codepath_v2/eligible_scope_manifest.tsv --out-dir data/results/20260415_targeted_core_a_repair_codepath_v2
python src/analysis/run_deterministic_step2_baseline_v1.py --step1-run-dir data/results/20260415_targeted_core_a_repair_codepath_v2 --table-row-binding-tsv data/results/20260415_targeted_core_a_repair_codepath_v2/table_row_binding_resolved_v1.tsv --parameter-binding-tsv data/results/20260415_targeted_core_a_repair_codepath_v2/formulation_parameter_binding_resolved_v1.tsv
```

## 6. Before -> After Metrics

Target papers only:

- EE-positive rows: `53 -> 53`
- Core Set A rows: `0 -> 32`
- `drug_name` support: `0 -> 53`
- `polymer_mw_kDa` support: `0 -> 32`
- `la_ga_ratio` support: `0 -> 32`
- remaining `candidate_absent_upstream`: `53 -> 21`
- remaining `ownership_unresolved`: `53 -> 0`
- repaired into Core Set A: `32`

By paper:

| paper_key | EE-positive before | EE-positive after | Core A before | Core A after |
|---|---:|---:|---:|---:|
| `WIVUCMYG` | 26 | 26 | 0 | 26 |
| `V99GKZEI` | 6 | 6 | 0 | 6 |
| `YGA8VQKU` | 16 | 16 | 0 | 0 |
| `BB3JUVW7` | 5 | 5 | 0 | 0 |

Detailed paper-level and row-level artifacts:

- `data/results/20260415_targeted_core_a_repair_codepath_v2/analysis/target_paper_core_a_metrics_v1.json`
- `data/results/20260415_targeted_core_a_repair_codepath_v2/analysis/target_paper_core_a_metrics_by_paper_v1.tsv`
- `data/results/20260415_targeted_core_a_repair_codepath_v2/analysis/target_paper_core_a_row_status_v1.tsv`

## 7. What Remains Unresolved

Remaining blocked EE-positive rows: `21`

- `YGA8VQKU` (`16` rows)
  - `drug_name` is now resolved
  - `polymer_mw_kDa` remains absent on the checked Core A path
  - `la_ga_ratio` remains absent on the checked Core A path
- `BB3JUVW7` (`5` rows)
  - `drug_name` is now resolved
  - EE-positive nanosphere rows still do not have lawful row-local `polymer_mw_kDa`
  - EE-positive nanosphere rows still do not have lawful row-local `la_ga_ratio`

No rows remained blocked by `ownership_unresolved` after the repair.

## 8. Is This Worth Extending?

Yes, with caution.

This repair shows that the dominant overlap loss was not caused by schema strictness. Narrow deterministic candidate recovery plus relation-backed ownership alignment produced a material gain (`0 -> 32` Core Set A rows) without changing formulation membership or `final_formulation_id`.

The next extension should stay equally narrow:

- inventory additional paper-local polymer-grade or ratio signals first
- add only evidence-backed target-paper recoveries
- do not generalize product-code parsing or family propagation beyond papers where the repo-local evidence is explicit enough to audit
