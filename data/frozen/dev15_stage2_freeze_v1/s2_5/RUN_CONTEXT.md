# RUN_CONTEXT

## 1. Run ID
`01_s2_5_semantic_parsing`

## 1a. Run Path
- run_dir: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\01_s2_5_semantic_parsing`
- run_dir_kind: `v2_child_execution`
- run_selection_mode: `default_v2_child`
- bucket_dir: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36`

## 2. Run Type
`intermediate_diagnostic_run`

Benchmark reporting rule:
- This run is `diagnostic-only, not benchmark-valid final output`.
- It materializes the frozen `S2-5` semantic-parsing surface only.
- It does not create the authoritative completed Stage2 artifact and it is not a lawful Stage3 resume boundary.

## 3. Purpose
- Consume frozen `S2-4b` raw-response payloads only.
- Parse those payloads into governed Stage2 semantic-intermediate object artifacts.
- Preserve object families, raw expressions, marker readiness, evidence handoff, and provenance-carrying intermediate structure.
- Stop before `S2-6` contract validation and before `S2-7` compatibility projection.

## 4. Stage Boundary
- current_stage_boundary: `S2-5`
- boundary_class: `internal_intermediate`
- authoritative_for_downstream: `no`
- lawful_resume_boundary: `no`
- stage_local_owner_script: `src/stage2_sampling_labels/run_stage2_s2_5_semantic_parsing_v1.py`
- stage_local_owner_function_surface:
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::convert_legacy_raw_response_to_v2`
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::normalize_replayed_live_document`
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::build_live_v2_document`
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::finalize_llm_first_document`
- stop_boundary: `semantic_intermediate_artifacts_written`
- next_lawful_step: `S2-6 contract validation`

## 5. Inputs
- manifest_tsv: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260401_5d9f4e6\08_dev15_count_validation\targeted_manifest.tsv`
- targeted_manifest_tsv: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\01_s2_5_semantic_parsing\targeted_manifest.tsv`
- raw_responses_dir: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260401_5d9f4e6\08_dev15_count_validation\semantic_stage2_objects\raw_responses`
- selected_paper_keys:
- `UFXX9WXE`
- `WIVUCMYG`
- selected_raw_response_files:
- `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260401_5d9f4e6\08_dev15_count_validation\semantic_stage2_objects\raw_responses\UFXX9WXE__stage2_v2_raw_response.json`
- `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260401_5d9f4e6\08_dev15_count_validation\semantic_stage2_objects\raw_responses\WIVUCMYG__stage2_v2_raw_response.json`
- input_contract_note:
  - frozen raw responses are the primary semantic input
  - manifest provenance is used only to resolve per-paper metadata, `text_path`, and table-file provenance
  - no clean text or evidence blocks are reread as the primary semantic input

## 6. Exact Script Execution Order
1. `src/stage2_sampling_labels/run_stage2_s2_5_semantic_parsing_v1.py`
2. `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::convert_legacy_raw_response_to_v2`
3. `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py::summary_row`

## 7. Outputs
- semantic-intermediate directory:
  - `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\01_s2_5_semantic_parsing\semantic_stage2_objects`
- semantic-intermediate JSONL:
  - `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\01_s2_5_semantic_parsing\semantic_stage2_objects\semantic_stage2_v2_objects.jsonl`
- semantic-intermediate summary TSV:
  - `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\01_s2_5_semantic_parsing\semantic_stage2_objects\semantic_stage2_v2_summary.tsv`
- run context:
  - `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\01_s2_5_semantic_parsing\RUN_CONTEXT.md`
- machine-readable run metadata:
  - `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260413_8517d36\01_s2_5_semantic_parsing\stage2_s2_5_run_metadata_v1.json`

## 8. Stop Rule
- This run stops after `S2-5` semantic-intermediate artifacts are written.
- The following substeps were not executed:
  - `S2-6 contract validation`
  - `S2-7 compatibility projection`
  - `Stage3`
  - `Stage4`
  - `Stage5`

## 9. Run Summary
- parsed_document_count: `2`
- failures: `0`
- success_status: `pass`
