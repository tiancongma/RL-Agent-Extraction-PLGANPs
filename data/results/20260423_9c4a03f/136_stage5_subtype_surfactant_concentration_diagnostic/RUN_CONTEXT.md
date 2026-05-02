# Stage5 subtype surfactant concentration diagnostic

- benchmark_valid: no
- compare_mode: diagnostic
- purpose: improve emulsifier/stabilizer concentration-value recall by source-backed nanocarrier subtype-scoped preparation carrythrough; suppress duplicate unit extras when GT stores full concentration in value field.
- upstream Stage2 compatibility: `data/results/20260423_9c4a03f/116_stage2_mass_carrythrough_derived_provenance_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- Stage3 relation dir: `data/results/20260423_9c4a03f/131_stage3_polymer_mass_scoped_preparation_repair_diagnostic`
- Stage5 final table: `data/results/20260423_9c4a03f/136_stage5_subtype_surfactant_concentration_diagnostic/final_formulation_table_v1.tsv`
- Layer3 compare output: `data/results/20260423_9c4a03f/138_layer3_compare_subtype_surfactant_concentration_unit_suppressed_diagnostic/layer3_value_compare_cells_v1.tsv`
- GT authority: `data/cleaned/gt_authority/v1/dev15_layer3_values.tsv`
- scope manifest: `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv`
- alignment scaffold: `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_variant_alignment_scaffold_v1.tsv`
- exact stage sequence: Stage2 compatibility from run 116 -> Stage3 131 -> Stage5 136 -> Layer3 compare 138.
- status: diagnostic-only lineage; not benchmark-valid final evidence.
- validation: tests `python3 -m unittest tests.test_compare_layer3_values_v1` passed (158 tests); py_compile passed for modified Stage5/compare/tests.
- delta vs run 135: overall error_rows 1773 -> 1757, present_and_match 1631 -> 1647, missing_in_system 886 -> 870, mismatch 85 unchanged, blocked_alignment 616 unchanged, extra_in_system 186 unchanged.
- target field delta: emulsifier_stabilizer_concentration_value recall 0.574324 -> 0.682432, match 85 -> 101, missing 49 -> 33, accuracy 1.000000 unchanged.
