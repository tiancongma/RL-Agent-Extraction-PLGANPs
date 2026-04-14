# RUN_CONTEXT

## 1. Run ID
`03_s2_7_compatibility_projection`

## 1a. Run Path
- run_dir: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\03_s2_7_compatibility_projection`
- run_dir_kind: `v2_child_execution`
- run_selection_mode: `default_v2_child`
- bucket_dir: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36`

## 2. Run Type
`intermediate_diagnostic_run`

Benchmark reporting rule:
- This run is `diagnostic-only, not benchmark-valid final output`.
- It materializes the frozen `S2-7` completed Stage2 artifact only.
- It does not execute `Stage3`, `Stage4`, or `Stage5`.

## 3. Purpose
- Consume the `S2-6`-validated `S2-5` semantic-intermediate surface only.
- Invoke the maintained compatibility-projection logic without any new LLM call.
- Materialize the completed Stage2 artifact required by unchanged downstream `Stage3` consumers.
- Stop before `Stage3` execution.

## 4. Stage Boundary
- current_stage_boundary: `S2-7`
- boundary_class: `mainline_resume_boundary`
- authoritative_for_downstream: `yes`
- lawful_resume_boundary: `yes`
- resume_entrypoint: `src/stage3_relation/run_formulation_relation_artifacts_v1.py`
- stage_local_owner_script: `src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py`
- stage_local_owner_function_surface:
  - `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py::run_projection`
  - `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py::project_document`
- stop_boundary: `completed_stage2_artifact_written`
- next_lawful_step: `Stage3 relation materialization`

## 5. Inputs
- source_s2_6_run_dir: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\02_s2_6_contract_validation`
- source_validation_report: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\02_s2_6_contract_validation\analysis\stage2_semantic_authority_contract_report_v1.json`
- source_validation_status: `pass`
- source_s2_5_run_dir: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\01_s2_5_semantic_parsing`
- semantic_jsonl: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\01_s2_5_semantic_parsing\semantic_stage2_objects\semantic_stage2_v2_objects.jsonl`
- input_contract_note:
  - this runner requires a passing `S2-6` validation report before projection
  - the semantic JSONL path is resolved from the `S2-6` validation surface
  - this runner does not consume raw `S2-4b` responses
  - this runner does not perform semantic parsing or contract validation
  - this runner does not invoke any live model call

## 6. Exact Script Execution Order
1. `src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py`
2. `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py::run_projection`

## 7. Outputs
- completed Stage2 artifact directory:
  - `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\03_s2_7_compatibility_projection\semantic_to_widerow_adapter`
- completed Stage2 weak-label TSV:
  - `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\03_s2_7_compatibility_projection\semantic_to_widerow_adapter\weak_labels__v7pilot_r3_fixparse.tsv`
- completed Stage2 weak-label JSONL:
  - `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\03_s2_7_compatibility_projection\semantic_to_widerow_adapter\weak_labels__v7pilot_r3_fixparse.jsonl`
- compatibility trace TSV:
  - `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\03_s2_7_compatibility_projection\semantic_to_widerow_adapter\compatibility_projection_trace_v1.tsv`
- compatibility summary JSON:
  - `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\03_s2_7_compatibility_projection\semantic_to_widerow_adapter\compatibility_projection_summary_v1.json`
- numbered DOE guard TSV:
  - `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\03_s2_7_compatibility_projection\semantic_to_widerow_adapter\numbered_doe_regression_guard_v1.tsv`
- projection contract TSV:
  - `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\03_s2_7_compatibility_projection\semantic_to_widerow_adapter\stage2_replacement_compatibility_projection_contract.tsv`
- run context:
  - `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\03_s2_7_compatibility_projection\RUN_CONTEXT.md`
- machine-readable run metadata:
  - `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\03_s2_7_compatibility_projection\stage2_s2_7_run_metadata_v1.json`

## 8. Stop Rule
- This run stops after `S2-7` compatibility projection artifacts are written.
- The following stages were not executed:
  - `Stage3`
  - `Stage4`
  - `Stage5`

## 9. Run Summary
- projected_document_count: `2`
- projected_row_count: `14`
- completed_stage2_artifact_status: `written`
- lawful_stage3_resume_boundary: `yes`

## Boundary Governance

- `boundary_class`: `mainline_resume_boundary`
- `authoritative_for_downstream`: `yes`
- `lawful_resume_boundary`: `yes`
- `resume_entrypoint`: `src/stage3_relation/build_formulation_relation_artifacts_v1.py`
- `schema_contract`: `completed_stage2_compatibility_projection_v1`
- `upstream_authority_source`: `unknown`
- `replay_mode`: `none`
- `source_run_dir`: `not_applicable`
- `source_run_id`: `not_applicable`
- `source_files`: `[]`
- `benchmark_terminal`: `no`
- `boundary_evidence_path`: `data/results/20260413_8517d36/03_s2_7_compatibility_projection/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`

## Feature Unit Activation

- `feature_activation_report_path`: `data/results/20260413_8517d36/03_s2_7_compatibility_projection/analysis/feature_activation_report_v1.tsv`
- `required_feature_units`: `["numbered_doe_row_enumeration_priority", "numbered_doe_regression_guard", "table_first_evidence_binding", "s2_candidate_section_aware_split", "s2_candidate_noise_filtering", "s2_2_evidence_artifact_contract", "s2_2_design_success_split", "s2_2_prompt_preview_derived_from_evidence_artifact", "s2_2_role_aware_evidence_selection", "s2_2_duplicate_table_suppression", "variant_aware_gt_authority_switch", "feature_unit_governance_layer", "run_context_feature_activation_integration"]`
- `missing_required_feature_units`: `["numbered_doe_row_enumeration_priority", "s2_candidate_section_aware_split", "s2_candidate_noise_filtering", "s2_2_evidence_artifact_contract", "s2_2_design_success_split", "s2_2_prompt_preview_derived_from_evidence_artifact", "s2_2_role_aware_evidence_selection", "s2_2_duplicate_table_suppression", "variant_aware_gt_authority_switch"]`
- `run_activation_gate`: `fail`

| feature_id | expected_for_run | observed_activation | activation_status | activation_state | evidence_path |
|---|---|---|---|---|---|
| `numbered_doe_row_enumeration_priority` | `yes` | `missing` | `missing` | `evidence_missing` | `data/results/20260413_8517d36/03_s2_7_compatibility_projection/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv` |
| `numbered_doe_regression_guard` | `yes` | `active` | `active` | `active` | `data/results/20260413_8517d36/03_s2_7_compatibility_projection/semantic_to_widerow_adapter/numbered_doe_regression_guard_v1.tsv` |
| `table_first_evidence_binding` | `yes` | `unclear` | `unclear` | `processing_error` | `data/results/20260413_8517d36/03_s2_7_compatibility_projection/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv` |
| `stage2_input_evidence_packing` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `s2_candidate_section_aware_split` | `yes` | `missing` | `missing` | `evidence_missing` | `` |
| `s2_candidate_table_isolation` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `s2_candidate_noise_filtering` | `yes` | `missing` | `missing` | `evidence_missing` | `` |
| `s2_2_evidence_artifact_contract` | `yes` | `missing` | `missing` | `evidence_missing` | `` |
| `s2_2_design_success_split` | `yes` | `missing` | `missing` | `evidence_missing` | `` |
| `s2_2_prompt_preview_derived_from_evidence_artifact` | `yes` | `missing` | `missing` | `evidence_missing` | `` |
| `s2_2_role_aware_evidence_selection` | `yes` | `missing` | `missing` | `evidence_missing` | `` |
| `s2_2_doe_overlay_selection` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `s2_2_duplicate_table_suppression` | `yes` | `missing` | `missing` | `evidence_missing` | `` |
| `stage2_json_sanitation_path1` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `stage2_legacy_fixparse_fallback_path2` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `variant_aware_gt_authority_switch` | `yes` | `missing` | `missing` | `evidence_missing` | `` |
| `family_variant_retention_governance` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
| `feature_unit_governance_layer` | `yes` | `active` | `active` | `active` | `data/results/20260413_8517d36/03_s2_7_compatibility_projection/analysis/feature_activation_report_v1.tsv` |
| `run_context_feature_activation_integration` | `yes` | `active` | `active` | `active` | `data/results/20260413_8517d36/03_s2_7_compatibility_projection/RUN_CONTEXT.md` |
| `doi_level_gt_vs_pred_count_audit` | `no` | `not_expected` | `not_expected` | `not_invoked` | `` |
