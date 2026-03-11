# Requirements

This document defines the stable project requirements that remain valid after
consolidating the historical `project_specification_UPDATED_*` files.

## Core Project Requirements

- The project must produce an auditable, reproducible PLGA formulation dataset.
- Final released outputs must remain tabular and traceable to source evidence.
- Filesystem contracts must stay stable; additive change is preferred over path churn.
- Canonical path resolution must be centralized in `src/utils/paths.py`.

## Data and Artifact Requirements

- Authoritative cleaned/index assets must remain uniquely identifiable.
- Run-scoped outputs must write under `data/results/<run_id>/...`.
- Cleaned dataset assets must write under `data/cleaned/<dataset_id>/...`.
- Manual benchmark and review artifacts must write under `data/cleaned/labels/manual/...`.
- Human-review artifacts must include DOI-level metadata.

## Pipeline Behavior Requirements

- Stage responsibilities must remain separated:
  - Stage2: semantic extraction
  - Stage4: evaluation, counting, audit
  - Stage5: schema/core projection and database-oriented outputs
- Default entrypoints must be taken from `project/ACTIVE_PIPELINE_RUNBOOK.md`, not inferred from filename similarity.
- Durable methodology changes must be recorded in `project/4_DECISIONS_LOG.md`.

## Consolidated Historical Specification Notes

The historical `project_specification_UPDATED_20260130_v5.txt`,
`project_specification_UPDATED_20260131_v6.txt`, and
`project_specification_UPDATED_20260201_v7.txt` repeatedly established these
requirements:

- authoritative data contracts must be explicit,
- run and cleaned assets must stay separated,
- human-edited artifacts must remain under manual-label locations,
- project governance should be carried by modular docs rather than a monolithic specification file.
