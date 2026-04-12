# RUN_CONTEXT

## 1. Run ID

- `run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1`

## 2. Run Type

- `full_pipeline_benchmark_experiment`

## 3. Purpose

- Run a DEV15-only controlled benchmark experiment that preserves Stage2-detected identity-bearing variables through the active semantic Stage2 -> compatibility adapter -> Stage3 -> Stage5 chain.
- Measure whether the additive identity-variable carrier reduces false collapse and improves agreement against the manually annotated final GT workbook authority:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.xlsx`

## 4. Starting Inputs

- Active baseline run resolved through:
  - `data/results/ACTIVE_RUN.json`
- Active baseline run directory:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1`
- Active DEV15 scope copied into this experiment:
  - `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv`
- Stage2 semantic source script:
  - `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`
- Stage2 compatibility adapter script:
  - `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
- Stage3 script:
  - `src/stage3_relation/build_formulation_relation_artifacts_v1.py`
- Stage5 script:
  - `src/stage5_benchmark/build_minimal_final_output_v1.py`
- Explicit GT authority for this experiment:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.xlsx`
- Baseline final table used for comparison:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/44_stage5_descendant_fix_v1fixed_recompare/run_20260321_1454_5fa3ed0_dev15_stage5_descendant_fix_v1fixed_recompare_v1/final_formulation_table_v1.tsv`

## 5. Exact Script Execution Order

1. Copy the active DEV15 scope manifest into the new experiment run directory.
2. Run the active semantic Stage2 emitter on the copied DEV15 scope.
3. Run the active Stage2 compatibility adapter to produce the legacy wide-row bridge surface with the additive `identity_variables_json` carrier.
4. Run the active Stage3 relation builder on the compatibility-projected weak-label surface.
5. Run the active Stage5 final-table builder on the Stage2/Stage3 artifacts.
6. Attempt the maintained compare script against the explicit repaired-v4 GT workbook.
7. Because the repaired-v4 workbook does not contain the `review_formulations` worksheet expected by the maintained compare script, run a read-only custom evaluator against the workbook's `value_gt_annotation` sheet to compare:
   - the active baseline final table
   - the new experiment final table
   against the same explicit manual GT authority.

## 6. Script Paths Used

- `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`
- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
- `src/stage3_relation/build_formulation_relation_artifacts_v1.py`
- `src/stage5_benchmark/build_minimal_final_output_v1.py`
- `src/stage5_benchmark/compare_final_table_to_gt_v1.py`

## 7. Commands Used

```powershell
$env:PYTHONPATH='c:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs'
$paperKeys = (Import-Csv 'data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv' -Delimiter "`t" | ForEach-Object { $_.key })
python src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py --manifest-tsv data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv --out-dir data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/semantic_stage2_objects --paper-keys $paperKeys
python src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py --input-jsonl data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/semantic_stage2_objects/semantic_stage2_objects_v1.jsonl --output-dir data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/semantic_to_widerow_adapter
python src/stage3_relation/build_formulation_relation_artifacts_v1.py --weak-labels-tsv data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv --weak-labels-jsonl data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.jsonl --scope-manifest-tsv data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv --out-dir data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/formulation_relation_v1
python src/stage5_benchmark/build_minimal_final_output_v1.py --input-tsv data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv --relation-records-tsv data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/formulation_relation_v1/formulation_relation_records_v1.tsv --resolved-relation-fields-tsv data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/formulation_relation_v1/resolved_relation_fields_v1.tsv --out-dir data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1
python src/stage5_benchmark/compare_final_table_to_gt_v1.py --final-table-tsv data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/final_formulation_table_v1.tsv --gt-xlsx data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.xlsx --scope-manifest-tsv data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv --out-dir data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/gt_manual_value_workbook_compare --scope-name dev15_identity_variable_preservation_exp
python src/stage5_benchmark/compare_final_table_to_gt_v1.py --final-table-tsv data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/44_stage5_descendant_fix_v1fixed_recompare/run_20260321_1454_5fa3ed0_dev15_stage5_descendant_fix_v1fixed_recompare_v1/final_formulation_table_v1.tsv --gt-xlsx data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.xlsx --scope-manifest-tsv data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/dev15_scope.tsv --out-dir data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/baseline_gt_manual_value_workbook_compare --scope-name dev15_active_baseline_vs_manual_value_workbook
```

## 8. Final Outputs

- `dev15_scope.tsv`
- `semantic_stage2_objects/semantic_stage2_objects_v1.jsonl`
- `semantic_stage2_objects/semantic_stage2_objects_summary_v1.json`
- `semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- `semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.jsonl`
- `semantic_to_widerow_adapter/compatibility_projection_contract_v1.tsv`
- `semantic_to_widerow_adapter/compatibility_projection_trace_v1.tsv`
- `semantic_to_widerow_adapter/compatibility_projection_summary_v1.json`
- `formulation_relation_v1/formulation_relation_records_v1.tsv`
- `formulation_relation_v1/formulation_logic_graph_v1.jsonl`
- `formulation_relation_v1/formulation_relation_summary_v1.tsv`
- `formulation_relation_v1/resolved_relation_fields_v1.tsv`
- `final_formulation_table_v1.tsv`
- `final_output_decision_trace_v1.tsv`
- `final_output_summary_v1.md`
- `RUN_CONTEXT.md`

## 9. Benchmark Contract

- This experiment executed the active DEV15 semantic Stage2 -> compatibility adapter -> Stage3 -> Stage5 chain before comparison.
- The explicit repaired-v4 manual workbook above is the GT authority for this experiment.
- The maintained compare script could not consume that workbook shape because it expects a `review_formulations` sheet. The benchmark comparison against this explicit GT authority therefore used a read-only custom evaluator after Stage5 rather than the maintained compare script outputs.
- The final-table outputs are benchmark-facing for this controlled DEV15 experiment, but the custom GT comparison should be interpreted together with the active baseline comparison because the semantic-emitter lineage materially changes row identity surfaces beyond the additive identity-variable carrier itself.
