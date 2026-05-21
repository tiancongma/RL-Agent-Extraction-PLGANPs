# Requirements

This document defines the stable project requirements that remain valid after
consolidating the historical `project_specification_UPDATED_*` files.

## Core Project Requirements

- The project must produce an auditable, reproducible PLGA formulation dataset.
- Final released outputs must remain tabular and traceable to source evidence.
- Filesystem contracts must stay stable; additive change is preferred over path churn.
- Canonical path resolution must be centralized in `src/utils/paths.py`.
- The system must preserve the division between LLM semantic discovery and
  deterministic validation, relation resolution, evidence binding, audit, and
  modeling-ready projection.

## Data and Artifact Requirements

- Authoritative cleaned/index assets must remain uniquely identifiable.
- Run-scoped outputs must write under `data/results/<run_id>/...`.
- Cleaned dataset assets must write under `data/cleaned/<dataset_id>/...`.
- Manual benchmark and review artifacts must write under `data/cleaned/labels/manual/...`.
- Human-review artifacts must include DOI-level metadata.
- PDF and HTML sources must be normalized into governed Stage1 document
  surfaces before downstream extraction. Downstream stages must consume
  declared text, structure, table, and sidecar interfaces rather than raw
  attachment formats directly.
- Stage1 interfaces must preserve row-level source provenance and must support
  scope overlays without changing the underlying corpus identity.
- Table-derived execution must use governed full-table authority surfaces when
  available, not lossy prompt summaries or ad hoc rereads of raw source files.

## Pipeline Behavior Requirements

- Stage responsibilities must remain separated:
  - Stage1: source normalization into unified document and table interfaces
  - Stage2: LLM-centered semantic extraction plus deterministic post-LLM
    completion into a downstream-ready artifact
  - Stage3: deterministic formulation relation materialization and resolution
  - Stage4: evaluation, counting, diagnostics, and reviewer-support surfaces
  - Stage5: fixed-row final-table closure, source-backed materialization,
    value-layer sidecars, audit export, and database-oriented outputs
- Default entrypoints must be taken from `project/ACTIVE_PIPELINE_RUNBOOK.md`, not inferred from filename similarity.
- Durable methodology changes must be recorded in `project/4_DECISIONS_LOG.md`.
- Prompt construction, live LLM calls, semantic parsing, contract validation,
  and compatibility projection must remain separate boundaries when the
  fine-grained Stage2 workflow is used.
- Deterministic table expansion may occur only after the LLM semantic layer has
  authorized the relevant table or formulation scope and the execution path can
  resolve back to preserved table authority.
- Stage5 value layers must not create or remove formulation rows. Residual LLM
  value extraction, authority validation, and derived-value computation must
  remain distinguishable from direct source-backed materialization.

## Cross-Cutting System Requirements

- Dictionary and normalization governance must support paper-local
  abbreviations, controlled vocabularies, curated promotion review, and
  downstream field harmonization without replacing Stage2 semantic discovery.
- Evidence binding must explain the source and assignment chain for frozen
  formulation rows and field values. It must not create rows, create values, or
  replace the final formulation table.
- Risk assessment must remain a review-prioritization sidecar over frozen
  extraction or evidence-binding artifacts. It must not re-resolve evidence or
  mutate binding facts.
- Human audit and reviewer workbook surfaces must preserve canonical Stage5
  formulation presence, source provenance, risk flags, and evidence-binding
  context. They must not redefine benchmark semantics.
- Modeling-ready outputs must be downstream projections from frozen curated or
  final-table surfaces. They may normalize, derive, and pivot values for
  modeling, but must preserve raw source-backed values and provenance.

## Diagnostic And Benchmark Boundary Requirements

- Intermediate artifacts, diagnostic reviews, risk outputs, and audit workbooks
  must not be reported as benchmark-valid final outputs.
- GT comparison must consume the completed Stage5 final formulation table and
  frozen GT authority as separate inputs.
- Current diagnostic baselines must be labeled as diagnostic unless a governed
  benchmark-valid contract explicitly says otherwise.

## Formulation Universe Discovery Requirements

- Formulation-row creation authority belongs to a controlled Stage2 semantic
  discovery gate before value binding. Downstream value-binding layers must not
  create formulation rows from numeric evidence.
- The gate may use multiple LLM passes, chunked source views, table snippets,
  adversarial review, and deterministic schema validation, but it must emit one
  stable, auditable formulation universe artifact per paper before downstream
  value extraction consumes it.
- The frozen universe artifact must record included prepared formulation
  instances, alias/identity rationale, row role, preparation evidence, excluded
  candidates with exclusion reasons, unresolved candidates for human review,
  source locators, model metadata, prompt hashes, and diagnostic-only status
  until explicitly promoted through the maintained Stage2 contract.
- Later value extraction may assign values only to frozen formulation IDs. If a
  later value pass detects a suspected missing formulation, it must write a
  review candidate instead of mutating the frozen row universe.

## Consolidated Historical Specification Notes

The historical `project_specification_UPDATED_20260130_v5.txt`,
`project_specification_UPDATED_20260131_v6.txt`, and
`project_specification_UPDATED_20260201_v7.txt` repeatedly established these
requirements:

- authoritative data contracts must be explicit,
- run and cleaned assets must stay separated,
- human-edited artifacts must remain under manual-label locations,
- project governance should be carried by modular docs rather than a monolithic specification file.
