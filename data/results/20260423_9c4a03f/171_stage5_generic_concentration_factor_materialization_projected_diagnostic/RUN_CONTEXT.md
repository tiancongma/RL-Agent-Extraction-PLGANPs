# Stage5 Generic Concentration Factor Materialization Diagnostic

- generated_at: `2026-05-03T14:51:50`
- benchmark_valid: `no`
- run_type: `diagnostic-only deterministic Stage5 replay`

## Inputs
- Stage2 weak labels: `data/results/20260423_9c4a03f/116_stage2_mass_carrythrough_derived_provenance_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- Stage3 relation records: `data/results/20260423_9c4a03f/146_stage3_generic_shared_parameter_bundle_diagnostic/formulation_relation_records_v1.tsv`
- Stage3 resolved relation fields: `data/results/20260423_9c4a03f/146_stage3_generic_shared_parameter_bundle_diagnostic/resolved_relation_fields_v1.tsv`

## Commands
```bash
python3 -m unittest tests.test_compare_layer3_values_v1
python3 -m py_compile src/stage5_benchmark/build_minimal_final_output_v1.py src/stage5_benchmark/compare_layer3_values_to_gt_v1.py tests/test_compare_layer3_values_v1.py
PYTHONPATH=. python3 src/stage5_benchmark/build_minimal_final_output_v1.py --input-tsv data/results/20260423_9c4a03f/116_stage2_mass_carrythrough_derived_provenance_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv --relation-records-tsv data/results/20260423_9c4a03f/146_stage3_generic_shared_parameter_bundle_diagnostic/formulation_relation_records_v1.tsv --resolved-relation-fields-tsv data/results/20260423_9c4a03f/146_stage3_generic_shared_parameter_bundle_diagnostic/resolved_relation_fields_v1.tsv --out-dir data/results/20260423_9c4a03f/171_stage5_generic_concentration_factor_materialization_projected_diagnostic
PYTHONPATH=. python3 src/stage5_benchmark/compare_layer3_values_to_gt_v1.py --final-table-tsv data/results/20260423_9c4a03f/171_stage5_generic_concentration_factor_materialization_projected_diagnostic/final_formulation_table_v1.tsv --decision-trace-tsv data/results/20260423_9c4a03f/171_stage5_generic_concentration_factor_materialization_projected_diagnostic/final_output_decision_trace_v1.tsv --layer3-gt-tsv data/cleaned/gt_authority/v1/dev15_layer3_values_gt_reviewed_split_units_v1.tsv --scope-manifest-tsv data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv --alignment-scaffold-tsv data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_variant_alignment_scaffold_v1.tsv --out-dir data/results/20260423_9c4a03f/172_layer3_compare_generic_concentration_factor_materialization_projected_diagnostic
```

## Repair Summary
- Added a bounded generic concentration-factor decoder for source-defined `X1`/`X2`/`X3` and `cMaterial` labels.
- The decoder only projects row-local assignments already present on admitted Stage5 rows; it does not create rows or expand DOE design space.
- Supported projections: drug concentration value/unit, polymer concentration value/unit, surfactant/stabilizer concentration text, and source-defined material names when uniquely bound.
- Generic compare surface now reads direct `drug_concentration_value/unit` columns instead of treating those fields as missing-system-field surfaces.

## Outputs
- `data/results/20260423_9c4a03f/171_stage5_generic_concentration_factor_materialization_projected_diagnostic/final_formulation_table_v1.tsv`
- `data/results/20260423_9c4a03f/171_stage5_generic_concentration_factor_materialization_projected_diagnostic/final_output_decision_trace_v1.tsv`
- `data/results/20260423_9c4a03f/172_layer3_compare_generic_concentration_factor_materialization_projected_diagnostic/layer3_value_compare_cells_v1.tsv`
- `analysis/layer3_field_repairs/run166_to_run172_generic_concentration_factor_delta_audit_v1.tsv`
- `analysis/layer3_field_repairs/run172_top3_paper_missing_boundary_audit_v1.tsv`

## Validation
- `python3 -m unittest tests.test_compare_layer3_values_v1`: 189 tests OK.
- `python3 -m py_compile ...`: passed.
- Diagnostic compare vs run166: error_rows `1603 -> 1559`; missing_in_system `836 -> 780`; present_and_match `1869 -> 1913`; present_but_mismatch `149 -> 161`; extra_in_system `30 -> 30`; blocked_alignment `588 -> 588`.

## Boundary / Validity
- This run is diagnostic-only, not benchmark-valid final output.
- Stage2 semantic authority was not replaced; the repair is a deterministic downstream materialization of source-defined coded factors within admitted row scope.
