# Stage5 P9 Polymer mg/mL Mass Guard Bounded Diagnostic

- generated_at: `2026-05-05T18:22:37`
- purpose: P9 bounded replay after auditing and repairing an overfill signal where row-local table headers such as `PLGA mg/mL` were being materialized as direct polymer mass.
- run_type: `bounded diagnostic replay`
- benchmark_valid: `no`
- boundary: lawful Stage5 input from completed Stage2 artifact (`029`) plus Stage3 relation artifacts (`030`); Layer3 compare uses Stage5 final table only.
- GT authority: `data/cleaned/gt_authority/v1/dev15_layer3_values_gt_source_completed_authority_v1.tsv`
- scope_manifest: `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv`
- source_run_dir resolved by compare: `/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260423_9c4a03f`
- note: diagnostic-only P9 validation; not a complete benchmark-valid final-output claim.

## Inputs

- weak_labels_tsv: `data/results/20260504_ab9f61e/029_stage2_current_baseline_replay_doe_explicit_only_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- relation_records_tsv: `data/results/20260504_ab9f61e/030_stage3_current_baseline_replay_doe_explicit_only_diagnostic/formulation_relation_records_v1.tsv`
- resolved_relation_fields_tsv: `data/results/20260504_ab9f61e/030_stage3_current_baseline_replay_doe_explicit_only_diagnostic/resolved_relation_fields_v1.tsv`

## Command

```bash
PYTHONPATH=. python3 src/stage5_benchmark/build_minimal_final_output_v1.py \
  --input-tsv data/results/20260504_ab9f61e/029_stage2_current_baseline_replay_doe_explicit_only_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv \
  --relation-records-tsv data/results/20260504_ab9f61e/030_stage3_current_baseline_replay_doe_explicit_only_diagnostic/formulation_relation_records_v1.tsv \
  --resolved-relation-fields-tsv data/results/20260504_ab9f61e/030_stage3_current_baseline_replay_doe_explicit_only_diagnostic/resolved_relation_fields_v1.tsv \
  --out-dir data/results/20260504_ab9f61e/049_stage5_p9_polymer_mgml_mass_guard_bounded_diagnostic
```

## Outputs

- final_table_tsv: `final_formulation_table_v1.tsv`
- decision_trace_tsv: `final_output_decision_trace_v1.tsv`
- downstream_variant_tsv: `downstream_variant_records_v1.tsv`
- summary_md: `final_output_summary_v1.md`
- summary: input_rows=252; final_rows=202; filtered_rows=46; collapsed_rows=4; kept_rows=202; downstream_variant_rows=5.

## Interpretation

Supersedes exploratory Stage5 P9 directories `040`, `044`, `045`, and `047`. The final generic guard does not create rows and does not use S5-3. It only blocks mass-field table-cell materialization when the raw header is a concentration header (`mg/mL`) for polymer mass aliases.
