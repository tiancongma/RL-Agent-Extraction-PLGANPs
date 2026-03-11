# Active Pipeline Runbook

This file is the current active script registry and execution-path guide for automated agents.

Use it to answer:
- which scripts are current default entrypoints,
- what they read and write,
- what order the active DEV-15 formulation-instance path follows,
- which nearby scripts are experimental, archival, or task-specific and should not be chosen by default.

## Recommended Read Order

1. [ACTIVE_PIPELINE_RUNBOOK.md](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/project/ACTIVE_PIPELINE_RUNBOOK.md)
2. [PIPELINE_SCRIPT_MAP.md](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/project/PIPELINE_SCRIPT_MAP.md)
3. [4_DECISIONS_LOG.md](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/project/4_DECISIONS_LOG.md)
4. [docs/tool_index.md](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/docs/tool_index.md)
5. Task-specific method notes under `docs/methods/`

## Status Vocabulary

- `ACTIVE_MAINLINE`: current default entrypoint for a real pipeline step
- `ACTIVE_SUPPORTING`: current supporting tool used around the mainline path
- `EXPERIMENTAL`: validated or useful experiment, but not the default entrypoint
- `ARCHIVAL`: kept for history, comparison, or backwards reference only

## Active Script Registry

### `src/utils/paths.py`

- `script_path`: `src/utils/paths.py`
- `status`: `ACTIVE_SUPPORTING`
- `purpose`: canonical repository path resolver
- `reads`: repository structure only
- `writes`: nothing
- `upstream`: none
- `downstream`: all stages
- `when_to_use`: whenever code needs project/data/run paths
- `when_not_to_use`: do not hardcode equivalent paths elsewhere

### `src/utils/run_preflight.py`

- `script_path`: `src/utils/run_preflight.py`
- `status`: `ACTIVE_SUPPORTING`
- `purpose`: run-id reuse/new-run preflight and model/input contract check
- `reads`: requested input paths and run metadata
- `writes`: run-preflight decision output, `runs/latest.txt` when used in run discipline
- `upstream`: user-selected task inputs
- `downstream`: any run-scoped script writing under `data/results/`
- `when_to_use`: before new run-scoped extraction/evaluation jobs
- `when_not_to_use`: not a data-processing step by itself

### `src/stage1_cleaning/zotero_raw_to_manifest.py`

- `script_path`: `src/stage1_cleaning/zotero_raw_to_manifest.py`
- `status`: `ACTIVE_SUPPORTING`
- `purpose`: build dataset manifest from raw Zotero-derived inputs
- `reads`: raw metadata and snapshot availability
- `writes`: cleaned dataset manifest TSVs
- `upstream`: Stage0 relevance/snapshot collection
- `downstream`: `clean_manifest_to_text.py`
- `when_to_use`: when building or refreshing a dataset manifest
- `when_not_to_use`: do not use for DEV-15 evaluation once the cleaned dataset manifests already exist

### `src/stage1_cleaning/clean_manifest_to_text.py`

- `script_path`: `src/stage1_cleaning/clean_manifest_to_text.py`
- `status`: `ACTIVE_SUPPORTING`
- `purpose`: generate cleaned text assets from a dataset manifest
- `reads`: dataset manifest TSV plus HTML/PDF source assets
- `writes`: `data/cleaned/<dataset_id>/text/...`
- `upstream`: `zotero_raw_to_manifest.py`
- `downstream`: Stage2 extraction and table extraction
- `when_to_use`: when cleaned text assets are missing or need regeneration
- `when_not_to_use`: do not rerun during Stage4-only evaluation/debug tasks

### `src/stage1_cleaning/run_tables_extraction_for_dataset_v1.py`

- `script_path`: `src/stage1_cleaning/run_tables_extraction_for_dataset_v1.py`
- `status`: `ACTIVE_SUPPORTING`
- `purpose`: build dataset-local table assets used by formulation-aware extraction and audit
- `reads`: cleaned dataset manifest plus source content
- `writes`: `data/cleaned/<dataset_id>/tables/...`
- `upstream`: cleaned dataset text/content
- `downstream`: Stage2 table-aware extraction and table audits
- `when_to_use`: when table assets are missing or stale
- `when_not_to_use`: not required for pure Stage4 counting over existing weak-label outputs

### `src/stage2_sampling_labels/sample_from_manifest_html_first.py`

- `script_path`: `src/stage2_sampling_labels/sample_from_manifest_html_first.py`
- `status`: `ACTIVE_SUPPORTING`
- `purpose`: reproducible subset or split creation from a cleaned manifest
- `reads`: dataset manifest TSV
- `writes`: sample/split manifest TSVs
- `upstream`: Stage1 cleaned manifest
- `downstream`: `build_key2txt_from_sample_manifest.py`, extraction scripts
- `when_to_use`: when defining or refreshing subset manifests
- `when_not_to_use`: do not use as a default extractor entrypoint

### `src/stage2_sampling_labels/build_key2txt_from_sample_manifest.py`

- `script_path`: `src/stage2_sampling_labels/build_key2txt_from_sample_manifest.py`
- `status`: `ACTIVE_SUPPORTING`
- `purpose`: build sample-local key-to-text index
- `reads`: sample/split manifest TSV
- `writes`: `key2txt.tsv`
- `upstream`: `sample_from_manifest_html_first.py`
- `downstream`: extraction/debug tools that need key-text lookup
- `when_to_use`: when a sample-local text index is required
- `when_not_to_use`: not needed for evaluation over already-produced weak-label TSVs

### `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`

- `script_path`: `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
- `status`: `ACTIVE_MAINLINE`
- `purpose`: current formulation-instance Stage2 pilot extractor with block-based evidence packing, synthesis-method priority, compressed instance enums, and table-heavy row-enumeration constraints
- `reads`: sample/split manifest TSV, cleaned text assets, optional table assets, model policy
- `writes`: run-scoped weak-label TSV and JSONL
- `upstream`: Stage1 cleaned text/tables plus split manifests
- `downstream`: `src/stage4_eval/eval_weak_labels_v7pilot3.py`
- `when_to_use`: current default Stage2 extractor for DEV-15 formulation-instance pilot work
- `when_not_to_use`: do not default to `auto_extract_weak_labels.py`, `auto_extract_weak_labels_v3.py`, `v4.py`, `v5.py`, `v5_G3.py`, or `v6.py` for current formulation-instance DEV work

### `src/stage2_sampling_labels/export_blockpack_audit_v7pilot_r3_fixparse.py`

- `script_path`: `src/stage2_sampling_labels/export_blockpack_audit_v7pilot_r3_fixparse.py`
- `status`: `ACTIVE_SUPPORTING`
- `purpose`: deterministic export of packed evidence blocks for human audit
- `reads`: cleaned text plus the current packer logic in `auto_extract_weak_labels_v7pilot_r3_fixparse.py`
- `writes`: block inventory TSV, packed order TSV, packed evidence text, audit note inputs
- `upstream`: Stage2 packer implementation
- `downstream`: manual packing review
- `when_to_use`: when debugging evidence packing without running the LLM
- `when_not_to_use`: not part of the default extraction/evaluation path

### `src/stage4_eval/eval_weak_labels_v7pilot3.py`

- `script_path`: `src/stage4_eval/eval_weak_labels_v7pilot3.py`
- `status`: `ACTIVE_MAINLINE`
- `purpose`: current Stage4 DEV evaluator for formulation-instance counting, paper-level diagnostics, and predicted-instance audit view
- `reads`: weak-label TSV, split manifest TSV, fixed GT workbook
- `writes`: `per_doi_formulation_instance_summary.tsv`, `predicted_instance_rows.tsv`, summary markdown
- `upstream`: `auto_extract_weak_labels_v7pilot_r3_fixparse.py` outputs
- `downstream`: combined DEV summary artifacts and manual review workbook
- `when_to_use`: current default Stage4 evaluator for DEV-15 formulation-instance work
- `when_not_to_use`: do not default to older alignment or multimodel scripts when the task is current DEV-15 formulation-instance counting

### `src/stage4_eval/build_dev15_review_workbook_v1.py`

- `script_path`: `src/stage4_eval/build_dev15_review_workbook_v1.py`
- `status`: `ACTIVE_SUPPORTING`
- `purpose`: build the manual DEV-15 Excel review workbook from evaluation summaries and predicted-instance rows
- `reads`: fixed GT workbook, combined DEV-15 eval TSV, per-DOI summaries, weak-label TSVs
- `writes`: `data/results/dev15_review/dev15_instance_review_v1.xlsx`
- `upstream`: `eval_weak_labels_v7pilot3.py` outputs plus combined summary artifact
- `downstream`: human review
- `when_to_use`: when preparing reviewer-facing DEV-15 inspection material
- `when_not_to_use`: not a counting or extraction entrypoint

### `src/archive_methods/benchmark_specific_audit_report/test_doe_coordinate_reconciliation_v1.py`

- `script_path`: `src/archive_methods/benchmark_specific_audit_report/test_doe_coordinate_reconciliation_v1.py`
- `status`: `ARCHIVAL`
- `purpose`: isolated validation script for DoE coordinate-signature merge logic
- `reads`: weak-label TSV for a target paper plus source text
- `writes`: experiment TSV summaries under `data/results/doe_coordinate_reconciliation_v1/`
- `upstream`: already-produced weak-label TSVs
- `downstream`: informed integration into `eval_weak_labels_v7pilot3.py`
- `when_to_use`: when validating or extending the coordinate-reconciliation rule on a small set of papers
- `when_not_to_use`: do not use as the default DEV evaluator; the validated logic is already integrated into `eval_weak_labels_v7pilot3.py`

### `src/stage5_benchmark/formulation_core_signature_v1.py`

- `script_path`: `src/stage5_benchmark/formulation_core_signature_v1.py`
- `status`: `ACTIVE_SUPPORTING`
- `purpose`: Stage5 formulation-core signature logic for database/core-level work
- `reads`: benchmark/schema-oriented formulation rows
- `writes`: core-signature outputs through its runner
- `upstream`: Stage5 benchmark/schema inputs
- `downstream`: Stage5 core projection/export
- `when_to_use`: when explicitly working on Stage5 core/schema outputs
- `when_not_to_use`: do not treat it as the current Stage4 DEV evaluator; the validated DoE reconciliation currently lives in `eval_weak_labels_v7pilot3.py`

## Current Active DEV-15 Path

This is the current formulation-instance DEV-15 path. Use these files and this order.

1. Fixed benchmark source artifacts
   - GT workbook:
     - `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v1_fixed.xlsx`
   - Split manifests:
     - `data/cleaned/goren_2025/index/splits/dev_manifest_v7pilot3_2026-03-06.tsv`
     - `data/cleaned/goren_2025/index/splits/dev_manifest_remaining12_2026-03-10.tsv`
   - Cleaned text assets:
     - `data/cleaned/goren_2025/text/...`
     - legacy-compatible content paths may also appear under `data/cleaned/content_goren_2025/text/...`

2. Stage2 extraction
   - Script:
     - `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
   - Main outputs:
     - run-scoped `weak_labels__v7pilot_r3_fixparse.tsv`
     - run-scoped `weak_labels__v7pilot_r3_fixparse.jsonl`

3. Stage4 evaluation
   - Script:
     - `src/stage4_eval/eval_weak_labels_v7pilot3.py`
   - Main outputs per evaluation run:
     - `per_doi_formulation_instance_summary.tsv`
     - `predicted_instance_rows.tsv`
     - summary markdown
   - Current validated DoE reconciliation location:
     - `src/stage4_eval/eval_weak_labels_v7pilot3.py`
   - Current validated DoE reconciliation case:
     - `WFDTQ4VX / 10.1080/10717544.2016.1199605`

4. Current official DEV-15 reconciled count view
   - Official current artifact:
     - `data/cleaned/labels/manual/formulation_instance_dev15_combined_eval_2026-03-10_reconciled.tsv`
   - Important:
     - this combined TSV is currently an assembled checked-in artifact derived from:
       - `data/cleaned/labels/manual/formulation_instance_remaining12_eval_2026-03-10_reconciled/per_doi_formulation_instance_summary.tsv`
       - `data/cleaned/labels/manual/formulation_instance_pilot3_eval_synthmethod_2026-03-10/per_doi_formulation_instance_summary.tsv`
     - there is not yet a dedicated checked-in canonical builder script for the full combined TSV

5. Manual review workbook
   - Script:
     - `src/stage4_eval/build_dev15_review_workbook_v1.py`
   - Output:
     - `data/results/dev15_review/dev15_instance_review_v1.xlsx`

## Stage-Specific Cautions

- Stage2 contains many extractor variants. For current DEV-15 formulation-instance work, only `auto_extract_weak_labels_v7pilot_r3_fixparse.py` is the default entrypoint.
- Stage4 contains many benchmark and diagnostic utilities. Most are useful for targeted analysis, not for default DEV counting.
- Stage5 contains schema/core tools. They are not the default evaluator path for the current DEV-15 formulation-instance benchmark.
- Do not assume a script is current because it has a higher suffix or a similar filename.

## Historical and Experimental Handling

Scripts not listed as `ACTIVE_MAINLINE` above must not be chosen as default entrypoints unless the task explicitly requires them.

Default non-entrypoint families for current formulation-instance DEV work:
- `src/archive_methods/older_weak_label_pilot_variants/auto_extract_weak_labels.py`
- `src/archive_methods/older_weak_label_pilot_variants/auto_extract_weak_labels_v3.py`
- `src/archive_methods/older_weak_label_pilot_variants/auto_extract_weak_labels_v4.py`
- `src/archive_methods/older_weak_label_pilot_variants/auto_extract_weak_labels_v5.py`
- `src/archive_methods/dual_model_extraction_comparison/auto_extract_weak_labels_v5_G3.py`
- `src/archive_methods/older_weak_label_pilot_variants/auto_extract_weak_labels_v6.py`
- `src/archive_methods/dual_model_extraction_comparison/auto_extract_multimodel.py`
- `src/archive_methods/dual_model_extraction_comparison/multi_model_extract_tier1.py`
- `src/archive_methods/dual_model_extraction_comparison/multi_model_extract_tier2.py`
- `src/archive_methods/dual_model_extraction_comparison/multi_model_merge_qc.py`
- `src/archive_methods/stage4_rule_heavy_formulation_reconstruction/apply_formulation_grouping_v1.py`
- `src/archive_methods/stage4_rule_heavy_formulation_reconstruction/apply_global_baseline_inheritance_and_rerun_alignment_v1.py`

If a task needs one of those scripts, treat it as explicit non-default work and state why.

---

## Consolidated Agent Execution Rules

This section consolidates the durable agent-governance content from the prior
agent-runbook files.

### Mandatory Preflight

Before any write-producing pipeline task, verify:

- `python` and `pip` resolve correctly,
- required dependencies import successfully,
- the repository root is the active working directory.

### Mandatory Read Order

Before changing pipeline behavior, read in this order:

1. `project/ACTIVE_PIPELINE_RUNBOOK.md`
2. `project/PIPELINE_SCRIPT_MAP.md`
3. `project/4_DECISIONS_LOG.md`
4. `project/2_ARCHITECTURE.md`
5. `docs/tool_index.md`
6. task-specific method docs when required

### Execution Invariants

- Use `src/utils/paths.py` for canonical path resolution.
- Do not infer the active mainline from filename similarity.
- Do not modify GT artifacts unless the task explicitly calls for GT maintenance.
- Do not write outputs under `src/`.

### Run Discipline

- Use run-preflight/run-id utilities for run-scoped work under `data/results/`.
- Keep run outputs, cleaned assets, and manual review artifacts in their designated roots.
- Update `project/4_DECISIONS_LOG.md`, `project/ACTIVE_PIPELINE_RUNBOOK.md`, and `project/PIPELINE_SCRIPT_MAP.md` when their respective governance scopes change.

### Legacy Compatibility Note

The former compatibility-pointer runbook existed only to preserve link stability.
Its role is now fully absorbed into this runbook.

