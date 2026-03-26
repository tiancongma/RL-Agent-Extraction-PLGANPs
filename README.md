# RL-Agent-Extraction-PLGANPs

This repository implements a governed, auditable pipeline for extracting
nanoparticle formulation records from literature and converting them into final
per-formulation structured outputs.

The active system is not just a prompt collection and not just a field
extraction stack. It is a Stage 0 to Stage 5 pipeline with explicit provenance:

1. Zotero-derived raw corpus intake
2. manifest and cleaned-content construction
3. semantic Stage2 object generation
4. deterministic compatibility projection into the legacy wide-row surface
5. deterministic formulation relation materialization
6. candidate-level diagnostics
7. final formulation-table closure and final-table benchmark comparison

The benchmark-valid system result is the Stage 5 final formulation table and its
GT-comparison outputs. Intermediate artifacts may be used for diagnosis, but
they are not official benchmark outputs.

The repository also supports a deterministic post-comparison Layer 2 risk
stratification artifact for downstream Layer 3 audit planning. That risk layer
is metadata only and does not alter benchmark-valid final outputs.

## Canonical Pipeline

The authoritative manual reproduction document is:

- [ACTIVE_PIPELINE_FLOW.md](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/project/ACTIVE_PIPELINE_FLOW.md)

That file defines the canonical Stage 0 to Stage 5 path, the exact stage
completion artifacts, the active scripts for each stage, and the allowed
incremental-reuse rules.

## Active Data Source Authority

Current `data/results` workflows must resolve their source artifacts through the
repository authority contract:

- [ACTIVE_DATA_SOURCE_CONTRACT.md](/c:/Users/tianc/Downloads/GitHub/RL-Agent-Extraction-PLGANPs/project/ACTIVE_DATA_SOURCE_CONTRACT.md)
- machine-readable pointer:
  - `data/results/ACTIVE_RUN.json`

The repository must not determine the active source by directory recency,
modification time, parent fallback, or glob-first matching. Use an explicit
CLI source such as `--run-dir`, or the declared authority pointer, or fail
loudly.

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

Stage 5 is a materialization layer. It must not perform semantic inference.
Benchmark-valid Stage 5 closure now depends on both Stage 3 relation records and
Stage 3 resolved relation fields.

Current Stage 5 supporting review export:

- `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py`
- `src/stage5_benchmark/build_field_gt_review_workbook_v1.py`

Current active Stage 3 script:

- `src/stage3_relation/build_formulation_relation_artifacts_v1.py`

Current active Stage 2 script:

- `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`

Current active compatibility bridge:

- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`

Current active Stage 4 diagnostic script:

- `src/stage4_eval/eval_weak_labels_v7pilot3.py`

Deprecated legacy Stage 2 fallback:

- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`

Current Layer 3 field-audit support surfaces:

- `src/stage5_benchmark/export_final_formulation_audit_ready_v1.py`
- `src/stage5_benchmark/build_field_gt_review_workbook_v1.py`
- `src/stage5_benchmark/build_layer2_risk_assessment_v1.py`

These are post-comparison review surfaces only. They do not alter frozen Stage
2, Stage 3, or Stage 5 semantics, and they do not change benchmark-valid final
outputs.

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
