# Source Code Overview

`src/` contains only active engineering code and stable stage-local tools.

The active stage namespaces are exactly:

- `src/stage0_relevance/`
- `src/stage1_cleaning/`
- `src/stage2_sampling_labels/`
- `src/stage3_gt/`
- `src/stage4_eval/`
- `src/stage5_benchmark/`
- `src/utils/`

Historical and retired methods do not live in `src/`. They live under
`archive/`.

## Stage 0

Purpose:
Relevance filtering and raw Zotero-derived corpus intake.

Key entrypoints:

- `zotero_api_sync_selected.py`
- `zotero_fetch_llm_relevant_pdfs.py`

## Stage 1

Purpose:
Manifest construction, cleaned text generation, and table extraction.

Key entrypoints:

- `zotero_raw_to_manifest.py`
- `clean_manifest_to_text.py`
- `run_tables_extraction_for_dataset_v1.py`

## Stage 2

Purpose:
Candidate formulation-instance extraction from cleaned paper assets.

Key entrypoint:

- `auto_extract_weak_labels_v7pilot_r3_fixparse.py`

## Stage 3

Purpose:
Checked manual benchmark assets.

Runtime note:

- no active routine stage-local Python script is required for ordinary runtime
- the canonical completion artifacts live under `data/cleaned/labels/manual/`

## Stage 4

Purpose:
Candidate-instance diagnostics and reviewer-facing audit surfaces.

Key entrypoints:

- `eval_weak_labels_v7pilot3.py`
- `build_dev15_review_workbook_v1.py`

## Stage 5

Purpose:
Final formulation-table closure and final-table benchmark comparison.

Key entrypoints:

- `build_minimal_final_output_v1.py`
- `compare_final_table_to_gt_v1.py`

Stage-local helper:

- `run_minimal_final_output_v1.py`

## Authoritative References

If script role or execution order is unclear, use:

- `project/ACTIVE_PIPELINE_FLOW.md`
- `project/PIPELINE_SCRIPT_MAP.md`
- `project/ACTIVE_PIPELINE_RUNBOOK.md`
