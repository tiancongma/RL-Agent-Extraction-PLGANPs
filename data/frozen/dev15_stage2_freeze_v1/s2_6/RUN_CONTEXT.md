# RUN_CONTEXT

## 1. Run ID
`02_s2_6_contract_validation`

## 1a. Run Path
- run_dir: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\02_s2_6_contract_validation`
- run_dir_kind: `v2_child_execution`
- run_selection_mode: `default_v2_child`
- bucket_dir: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36`

## 2. Run Type
`intermediate_diagnostic_run`

Benchmark reporting rule:
- This run is `diagnostic-only, not benchmark-valid final output`.
- It materializes the frozen `S2-6` contract-validation surface only.
- It does not create the authoritative completed Stage2 artifact and it is not a lawful Stage3 resume boundary.

## 3. Purpose
- Consume `S2-5` semantic-intermediate artifacts only.
- Validate semantic-source mode declaration, semantic scope provenance, marker readiness governance, and maintained semantic-intermediate contract compliance.
- Stop before `S2-7` compatibility projection.

## 4. Stage Boundary
- current_stage_boundary: `S2-6`
- boundary_class: `internal_intermediate`
- authoritative_for_downstream: `no`
- lawful_resume_boundary: `no`
- stage_local_owner_script: `src/stage2_sampling_labels/run_stage2_s2_6_contract_validation_v1.py`
- stage_local_owner_function_surface:
  - `src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py::validate_semantic_documents`
  - `src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py::validate_selection_marker`
  - `src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py::validate_inheritance_marker`
  - `src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py::has_llm_declared_doe_scope`
  - `src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py::has_table_formulation_scope_marker`
- stop_boundary: `semantic_contract_validation_artifacts_written`
- next_lawful_step: `S2-7 compatibility projection`

## 5. Inputs
- source_s2_5_run_dir: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\01_s2_5_semantic_parsing`
- semantic_jsonl: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\01_s2_5_semantic_parsing\semantic_stage2_objects\semantic_stage2_v2_objects.jsonl`
- semantic_summary_tsv: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\01_s2_5_semantic_parsing\semantic_stage2_objects\semantic_stage2_v2_summary.tsv`
- targeted_manifest_tsv: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\01_s2_5_semantic_parsing\targeted_manifest.tsv`
- source_s2_5_metadata: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\01_s2_5_semantic_parsing\stage2_s2_5_run_metadata_v1.json`
- input_contract_note:
  - the frozen `S2-5` semantic JSONL is the only required semantic input
  - this runner does not consume raw `S2-4b` responses
  - this runner does not reread clean text or evidence blocks as the primary semantic input
  - this runner does not invoke any live model call

## 6. Exact Script Execution Order
1. `src/stage2_sampling_labels/run_stage2_s2_6_contract_validation_v1.py`
2. `src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py::validate_semantic_documents`

## 7. Outputs
- validation report:
  - `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\02_s2_6_contract_validation\analysis\stage2_semantic_authority_contract_report_v1.json`
- run context:
  - `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\02_s2_6_contract_validation\RUN_CONTEXT.md`
- machine-readable run metadata:
  - `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\02_s2_6_contract_validation\stage2_s2_6_run_metadata_v1.json`

## 8. Stop Rule
- This run stops after `S2-6` validation artifacts are written.
- The following substeps were not executed:
  - `S2-5 semantic parsing`
  - `S2-7 compatibility projection`
  - `Stage3`
  - `Stage4`
  - `Stage5`

## 9. Run Summary
- declared_mode: `llm_first_composite`
- document_count: `2`
- warning_count: `0`
- error_count: `0`
- allowed_modes:
  - `['diagnostic_comparator', 'governed_fallback_semantic_source', 'llm_first_composite']`
- success_status: `pass`
