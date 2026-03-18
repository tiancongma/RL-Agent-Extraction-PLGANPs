# RL-Agent-Extraction-PLGANPs

This repository implements a governed, auditable pipeline for extracting
nanoparticle formulation records from literature and converting them into final
per-formulation structured outputs.

The active system is not just a prompt collection and not just a field
extraction stack. It is a Stage 0 to Stage 5 pipeline with explicit provenance:

1. Zotero-derived raw corpus intake
2. manifest and cleaned-content construction
3. candidate formulation-instance extraction
4. deterministic formulation relation materialization
5. candidate-level diagnostics
6. final formulation-table closure and final-table benchmark comparison

The benchmark-valid system result is the Stage 5 final formulation table and its
GT-comparison outputs. Intermediate artifacts may be used for diagnosis, but
they are not official benchmark outputs.

## Canonical Pipeline

The authoritative manual reproduction document is:

- [ACTIVE_PIPELINE_FLOW.md](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/project/ACTIVE_PIPELINE_FLOW.md)

That file defines the canonical Stage 0 to Stage 5 path, the exact stage
completion artifacts, the active scripts for each stage, and the allowed
incremental-reuse rules.

## Active Stage Namespaces

The active runtime stage directories are:

- `src/stage0_relevance/`
- `src/stage1_cleaning/`
- `src/stage2_sampling_labels/`
- `src/stage3_relation/`
- `src/stage4_eval/`
- `src/stage5_benchmark/`

The reserved non-runtime reference namespace is:

- `src/stage3_gt/`

There is no active Stage 6 or Stage 7 namespace. Historical or retired methods
live outside `src/` under `archive/`.

## Repository Layout

- `project/`
  Governance and authoritative pipeline definitions.
- `src/`
  Active engineering code and stable stage-local tools only.
- `archive/`
  Historical methods, retired code, and delete-candidate quarantine.
- `data/`
  Raw inputs, cleaned assets, manual labels, and run artifacts.
- `docs/`
  Supporting documentation, audits, registries, and governance support files.

## Current Runtime Contract

The canonical path starts from Zotero-derived raw records under
`data/raw/zotero/` and ends at:

- `final_formulation_table_v1.tsv`
- `final_table_vs_gt_counts.tsv`
- `final_table_vs_gt_summary.md`

Current active Stage 5 scripts:

- `src/stage5_benchmark/build_minimal_final_output_v1.py`
- `src/stage5_benchmark/compare_final_table_to_gt_v1.py`

Current Stage 5 supporting review export:

- `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py`

Current active Stage 3 script:

- `src/stage3_relation/build_formulation_relation_artifacts_v1.py`

Current active Stage 2 script:

- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`

Current active Stage 4 diagnostic script:

- `src/stage4_eval/eval_weak_labels_v7pilot3.py`

## Governance

Before changing code or pipeline documentation, read:

- [AGENTS.md](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/AGENTS.md)
- [2_ARCHITECTURE.md](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/project/2_ARCHITECTURE.md)
- [PIPELINE_SCRIPT_MAP.md](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/project/PIPELINE_SCRIPT_MAP.md)
- [ACTIVE_PIPELINE_FLOW.md](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/project/ACTIVE_PIPELINE_FLOW.md)
- [ACTIVE_PIPELINE_RUNBOOK.md](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/project/ACTIVE_PIPELINE_RUNBOOK.md)

## Reproducibility

Every `data/results/run_*` directory must contain a reproducibility-grade
`RUN_CONTEXT.md`.

Accepted run types are:

- `intermediate_diagnostic_run`
- `component_regression_run`
- `full_pipeline_benchmark_run`

Only `full_pipeline_benchmark_run` may report official GT results, and only
when the evaluation object is the Stage 5 final formulation table.
