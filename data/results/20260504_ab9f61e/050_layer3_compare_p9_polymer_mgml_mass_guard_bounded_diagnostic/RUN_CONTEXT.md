# Layer3 Compare Run

- generated_at: `2026-05-05T18:22:37`
- source_run_dir: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f`
- final_table_tsv: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260504_ab9f61e/049_stage5_p9_polymer_mgml_mass_guard_bounded_diagnostic/final_formulation_table_v1.tsv`
- layer3_gt_tsv: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/cleaned/gt_authority/v1/dev15_layer3_values_gt_source_completed_authority_v1.tsv`
- scope_manifest_tsv: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv`
- alignment_scaffold_tsv: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_variant_alignment_scaffold_v1.tsv`
- trusted_alignment_tsv: ``
- cells_tsv: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260504_ab9f61e/050_layer3_compare_p9_polymer_mgml_mass_guard_bounded_diagnostic/layer3_value_compare_cells_v1.tsv`
- summary_tsv: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260504_ab9f61e/050_layer3_compare_p9_polymer_mgml_mass_guard_bounded_diagnostic/layer3_value_compare_summary_v1.tsv`
- error_buckets_tsv: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260504_ab9f61e/050_layer3_compare_p9_polymer_mgml_mass_guard_bounded_diagnostic/layer3_value_error_buckets_v1.tsv`
- alignment_resolution_tsv: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260504_ab9f61e/050_layer3_compare_p9_polymer_mgml_mass_guard_bounded_diagnostic/layer3_alignment_resolution_rows_v1.tsv`
- risk_review_queue_tsv: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260504_ab9f61e/050_layer3_compare_p9_polymer_mgml_mass_guard_bounded_diagnostic/layer3_risk_review_queue_v1.tsv`

## P9 validation audit

- benchmark_valid: `no` (bounded diagnostic replay only).
- Baseline compare: `032_layer3_compare_current_s5_2_no_s5_3_baseline_doe_explicit_only_diagnostic`.
- Initial P9 compare before guard: `041_layer3_compare_generic_material_value_binding_p9_bounded_diagnostic`.
- Final P9 compare after guard: this directory (`050`).
- Audit sidecar: `p9_validation_audit_summary_v1.json`.

### Key result

- `041` showed a regression signal: new `extra_in_system` vs `032` = `{"polymer_mass_mg": 26}`, caused by `PLGA mg/mL` concentration headers being interpreted as direct polymer masses.
- Generic repair applied: `mg/mL` polymer mass headers are rejected for direct mass materialization; no paper-key branch or GT lookup was used.
- `050` removes that overfill: new `extra_in_system` vs `032` = `{}`; `polymer_mass_mg` extra remains `0` in core summary.
- Legitimate improvement retained: `drug_mass_mg` transitioned `missing_in_system -> present_and_match` for 6 cells with typed direct mass evidence.

### Core fixed field totals

- `032` core total: gt_nonempty=`3119`, system_on_gt=`2330`, recall=`0.747034`, extra=`34`.
- `041` initial P9: gt_nonempty=`3119`, system_on_gt=`2336`, recall=`0.748958`, extra=`60`.
- `050` final P9: gt_nonempty=`3119`, system_on_gt=`2336`, recall=`0.748958`, extra=`34`.

### Field checks

- `polymer_mass_mg`: baseline system_on_gt=`57`, final system_on_gt=`57`, baseline extra=`0`, final extra=`0`.
- `drug_mass_mg`: baseline system_on_gt=`27`, final system_on_gt=`33`, baseline extra=`0`, final extra=`0`.

### Conclusion

P9 bounded diagnostic validation passes after the generic polymer `mg/mL` mass-header guard: the overfill regression observed in `041` is resolved, final row count remains 202, core recall improves from `0.747034` to `0.748958`, and core extra remains unchanged at 34. This is diagnostic-only, not benchmark-valid final output.
