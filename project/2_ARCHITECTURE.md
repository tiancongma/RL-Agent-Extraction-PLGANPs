# Architecture

This document defines the conceptual and data architecture of the project.
It specifies the stable structure of the pipeline, the responsibilities of each
stage, and the data contracts between stages.

This file is intentionally conservative and should change only when the project
scope changes, not when individual prompts, models, or one-off experiments
change.

---

## Design Philosophy

This project is organized around data semantics, not script order.

- Scripts may change.
- Prompts may change.
- Models may change.

However, the data boundaries, directory semantics, and single sources of truth
must remain stable.

The goal is to ensure that:

- results are reproducible,
- experimental variation is explicit,
- "latest" is never inferred from filenames or memory.

---

## Pipeline Overview

The active implementation is restricted to Stage 0 through Stage 5.
Stages are conceptual contracts first and script namespaces second.

The canonical production path is:

1. Stage 0: raw metadata and relevance filtering
2. Stage 1: cleaned content and manifest construction
3. Stage 2: LLM candidate formulation-instance extraction
4. Stage 3: deterministic formulation relation materialization
5. Stage 4: evaluation and diagnostics
6. Stage 5: final formulation closure and benchmark comparison

Manual GT assets are reference inputs to evaluation and comparison. They are
not a production transformation stage.

---

## Stage 0 - Raw Metadata and Relevance Filtering

### Purpose
Identify candidate papers that are potentially relevant to PLGA nanoparticle
formulation and build the checked raw Zotero-derived input surface.

### Typical Operations
- regex-based pre-filtering
- LLM-based relevance classification
- Zotero tagging and attachment fetching

### Characteristics
- outputs are rerunnable
- outputs are non-binding
- downstream stages should depend on the checked raw JSONL artifact, not on
  transient helper state

### Location
`data/raw/zotero/`

---

## Stage 1 - Cleaned Content and Manifest

### Purpose
Convert HTML/PDF documents into cleaned text and stable corpus assets, then
establish a manifest linking papers to those assets.

### Key Outputs
- cleaned full text
- key-to-text mappings
- dataset-local table assets
- manifest rows linking key, DOI, title, and content paths

### Single Source of Truth
`data/cleaned/index/manifest_current.tsv`

### Location
- `data/cleaned/content/`
- `data/cleaned/index/`
- dataset-local cleaned roots such as `data/cleaned/goren_2025/`

### Invariants
- Any change to files in `data/cleaned/index/` requires re-running all
  downstream stages.
- Only one `manifest_current.tsv` may exist as the active authoritative manifest.

---

## Stage 2 - LLM Candidate Formulation-Instance Extraction

### Purpose
Generate high-recall candidate formulation-instance rows from cleaned paper
content using the active extraction schema, with additive deterministic recovery
of explicit numbered DOE table rows when Stage1 table assets contain a
numbered design or formulation table.

### Key Artifacts
- run-scoped weak-label TSVs
- run-scoped weak-label JSONL records
- run-scoped raw LLM responses
- additive deterministic Stage2 augmentation artifacts for numbered DOE tables:
  - `numbered_doe_row_candidates_v1.tsv`
  - `numbered_doe_row_candidates_summary_v1.tsv`

### Characteristics
- this is the semantic extraction layer
- instance boundaries and field-role interpretation are assigned here
- outputs are intentionally high recall and may contain overlapping candidates
- deterministic numbered DOE row recovery is allowed here because downstream
  deterministic stages cannot reconstruct explicit numbered rows that the LLM
  never emitted

### Canonical Completion Artifact
`data/results/run_<run_id>/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`

---

## Stage 3 - Deterministic Formulation Relation Materialization

### Purpose
Convert Stage 2 candidate formulation-instance rows into explicit paper-level
relation artifacts without any LLM usage.

### Why this stage exists
The pipeline needs an auditable intermediate layer that makes relation
structure explicit before final flattening. This layer separates relation
reasoning from final benchmark-facing row closure.

### Key Artifacts
- relation-record TSVs
- per-paper logic-graph JSON artifacts
- per-paper relation summary TSVs

### Characteristics
- deterministic and reproducible
- no LLM or external API calls
- explicit method-group, shared-field, variation-axis, and parent-link structure
- intermediate production artifact, not benchmark-valid output

### Location
- `src/stage3_relation/`
- `data/results/run_<run_id>/...`

---

## Stage 4 - Evaluation and Diagnostics

### Purpose
Quantitatively evaluate extraction quality and support targeted debugging and
review.

### Typical Operations
- rule-based grading
- metric calculation
- reviewer-facing mismatch and alignment surfaces

### Characteristics
- downstream of Stage 2 candidate extraction
- diagnostic only
- not the benchmark-valid system endpoint

### Location
`src/stage4_eval/`

---

## Evaluation Reference Assets

### Purpose
Provide partial, human-curated labels for comparison and review.

### Characteristics
- ground truth is intentionally incomplete
- manual labels do not overwrite weak labels
- disagreement between system output and GT is expected during development

### Location
`data/cleaned/labels/manual/`

---

## Stage 5 - Final Formulation Closure And Benchmark Comparison

### Purpose
Convert candidate formulation-instance outputs into final one-row-per-
formulation records and compare only those final records to GT.

### Key Principle
Stage 5 is the only benchmark-valid reporting layer. Earlier stages may produce
diagnostic comparisons, but they are not the official system result.

### Current Phase-1 Boundary
- Stage 5 consumes the Stage 2 candidate TSV directly.
- It may also consume the Stage 3 relation-record TSV as deterministic
  provenance.
- The relation artifact is not yet the sole driver of final keep/drop/collapse
  decisions.

### Location
- `src/stage5_benchmark/`
- `data/results/run_<run_id>/`

---

## Run Tracking and Reproducibility

Each pipeline execution is assigned a unique `run_id`:

`run_YYYYMMDD_HHMM_<git_commit>_<sample>`

Run outputs live under:

`data/results/run_<run_id>/`

Every run directory must contain a reproducibility-grade `RUN_CONTEXT.md`.

### Lineage containment policy

A top-level `data/results/run_*` directory now represents one benchmark or
experiment lineage, not every internal retry or repair step.

Child executions that belong to the same lineage must live under the parent
run directory, for example:

- `data/results/<parent_run_id>/lineage/children/<ordered_role>/<child_run_id>/`

This child-execution rule applies to:

- stage-local retries
- partial reruns
- recovery passes for failed papers
- deterministic refresh steps
- stage-only materialization runs
- merge or completion steps that serve the same declared lineage objective

The parent lineage directory remains the authoritative human-facing entrypoint.
It must expose:

- its own `RUN_CONTEXT.md`
- a lineage mapping artifact if child runs were moved or nested
- an explicit child-step index when the lineage includes retries or repair work

Independent top-level runs are allowed only when the declared objective, scope,
or benchmark contract is materially separate from an existing lineage.

---

## Architectural Invariants

- Files in `data/cleaned/index/` are pipeline-critical.
- `manifest_current.tsv` and `key2txt.tsv` are unique active authorities.
- No script may silently depend on legacy artifacts.
- Experimental variation must be expressed via:
  - `run_id`
  - explicit configuration or CLI arguments
  - git commit
- Official benchmark reporting may occur only from Stage 5 final-table
  comparison outputs.

---

## Relationship to Project Governance

- project scope and stage transitions are defined here
- architectural decisions are recorded in `project/4_DECISIONS_LOG.md`
- agent run procedure is defined in `project/ACTIVE_PIPELINE_RUNBOOK.md`
- active script registry and stage roles are defined in
  `project/PIPELINE_SCRIPT_MAP.md` and `docs/src_script_registry.tsv`

This architecture is intentionally minimal, explicit, and extensible.

---

## Debug And Human Review Contract

All debug artifacts intended for manual inspection must include DOI-level
metadata.

Any Excel or TSV generated for human review must contain:

- doc_key (Zotero key)
- reference_normalized_doi (DOI)
- doi_url (`https://doi.org/<DOI>`)
- paper_title when available
- publication_year when available

Rationale:

- manual verification requires immediate access to original publications
- DOI is the canonical external identifier and must be included

Scope:

- benchmark debug matrices
- patch-queue outputs
- per-document regression diagnostics
- modeling-ready merged instance summaries
- any artifact labeled debug, audit, review, or manual

No debug artifact is considered complete if DOI is missing.

---

## LLM Extraction Layer, Deterministic Arbitration Layer, And Audit Boundary

### Why the pipeline is layered
The project separates semantic extraction, deterministic arbitration, and audit
so that:

- semantic interpretation is handled where language context is strongest
  (LLM stage),
- reproducible rule behavior is handled where strict consistency is required
  (deterministic stages),
- human and machine audit can verify both without mixing responsibilities.

This prevents hidden logic drift where semantic decisions are silently embedded
into late-stage scripts.

### Layer 1: LLM extraction responsibilities
The LLM extraction layer is responsible for:

- identifying formulation instances and instance boundaries
- assigning field-role semantics
- distinguishing shared-vs-instance-specific meaning in prose and tables
- emitting structured candidate rows with explicit missingness rather than
  silent omission

The LLM extraction layer is not responsible for final arbitration of
conflicting evidence.

### Layer 2: Deterministic arbitration responsibilities
Deterministic scripts are responsible for:

- formulation relation materialization and grouping provenance
- numeric evidence binding and token-level support checks
- deterministic derivation and unit or ratio normalization
- schema assembly and export formatting
- stable filtering and gating for reproducible benchmarking and release outputs

These responsibilities must remain deterministic to preserve run-to-run
reproducibility and auditability.

### Layer 3: Audit boundary responsibilities
Audit occurs at the boundary between extracted candidates and publishable
database outputs. The audit boundary must expose:

- evidence pointers and span traceability
- field-level and formulation-level QC outcomes
- explicit conflict and uncertainty artifacts for targeted human review

### Stable downstream deterministic rule families
The following rule families are considered stable deterministic core:

- formulation relation materialization
- numeric evidence realignment and token QC gating
- derivation and normalized field computation
- schema assembly and export formatting
- benchmark-facing final-table comparison
