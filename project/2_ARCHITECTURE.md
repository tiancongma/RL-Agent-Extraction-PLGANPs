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
3. Stage 2: composite semantic extraction and deterministic post-LLM completion
4. Stage 3: deterministic formulation relation materialization
5. Stage 4: evaluation and diagnostics
6. Stage 5: final formulation closure and benchmark comparison

Manual GT assets are reference inputs to evaluation and comparison. They are
not a production transformation stage.

Architecture note:

- Frozen corrective contract after the 2026-03-30 Stage2 authority-transition
  audit:
  - Stage2 authority belongs to LLM semantic discovery, not deterministic
    semantic reconstruction.
  - Deterministic Stage2 semantic emitters or semantic lifts are fallback,
    comparator, migration-support, or diagnostic infrastructure only.
  - Future drift that re-promotes deterministic semantic Stage2 authority
    should be treated as a contract violation.
- The deterministic post-LLM completion step remains inside Stage2. It does not
  own semantic discovery authority and it is not a separate numbered stage.
- Stage2.5 is retired from the active mainline and remains archived only as a
  historical exploratory path.

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

## Stage 2 - Composite Semantic Extraction Layer

### Purpose
Run one composite Stage2 contract over cleaned paper content and structured
assets:

1. LLM semantic discovery
2. deterministic post-LLM completion for downstream readiness

Stage2 is the authoritative extraction stage. Raw LLM semantic objects are an
internal Stage2 intermediate. The completed Stage2 artifact is the only valid
Stage3 input and the only authoritative Stage2 evaluation target.

### Key Artifacts
- Stage2 internal semantic-intermediate artifacts:
  - run-scoped semantic-object JSONL payloads
  - run-scoped semantic-object summary TSVs
  - run-scoped raw response copies when replay or live LLM execution is used
  - paper-local evidence handoff references carried inside the semantic objects
  - object families:
  - `formulation_identity_candidate`
  - `component_candidate`
  - `phase_candidate`
  - `process_step_candidate`
  - `variable_or_factor_candidate`
  - `measurement_candidate`
  - `relation_cue`
  - `evidence_handoff`
- Stage2 completed downstream-ready artifacts:
  - compatibility-projected legacy wide-row TSV
  - compatibility-projected legacy wide-row JSONL
  - projection trace TSV
  - projection summary JSON

### Characteristics
- Stage2 is one composite stage, not multiple numbered stages
- open semantic discovery and formulation-boundary discovery are owned by the
  LLM substep
- deterministic post-LLM completion remains inside Stage2 and exists only to
  make Stage2 outputs reconstructable and relation-ready for unchanged
  downstream consumers
- formulation identity discovery, component discovery, factor discovery, and
  raw expression capture are owned by the LLM substep
- raw semantic objects may remain incomplete where the paper support is
  incomplete
- deterministic post-LLM completion must not be mistaken for Stage3 or Stage5
- Stage2 final output must not be treated as final benchmark materialization
- deterministic Stage2 semantic reconstruction paths are non-authoritative and
  must not replace the LLM Stage2 boundary as active mainline authority
- direct comparison of raw semantic objects to formulation-level GT is
  diagnostic only when the deterministic completion substep has not been
  applied

### Stage2 Internal Intermediate Artifact
`data/results/run_<run_id>/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`

### Stage2 Authoritative Completion Artifact
`data/results/run_<run_id>/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`

---

## Stage 3 - Deterministic Formulation Relation Materialization

### Purpose
Convert the completed Stage2 candidate formulation-instance rows into explicit
paper-level relation artifacts without any LLM usage.

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
- downstream of the compatibility-projected legacy wide-row surface
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

---

## Separation Of Concerns

- Stage2 owns semantic discovery from paper text and tables plus the
  deterministic post-LLM completion required for downstream readiness.
- Stage3 owns relation resolution over the compatibility-projected rows.
- Stage5 owns final materialization and benchmark-facing closure.
- Stage5 must not absorb semantic inference that belongs to Stage2.

### Current Phase-1 Boundary
- Stage 5 consumes the compatibility-projected legacy wide-row TSV produced by
  the deterministic adapter.
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

### Active data-source authority

For current `data/results` workflows, the repository-level active source must
be declared explicitly.

Authority order:

1. explicit CLI source such as `--run-dir`
2. `data/results/ACTIVE_RUN.json`
3. otherwise hard error

The architecture forbids resolving the active source by:

- lexical sort order
- modification time
- parent fallback
- glob-first matching
- unstated defaulting

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
- emitting flexible semantic objects rather than relying on deterministic
  semantic reconstruction as the active authority

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

Stage2 enforcement note:

- Deterministic semantic emitters, deterministic semantic lifts, and similar
  rule-heavy Stage2 reconstruction paths may exist for fallback, comparator,
  migration-support, or diagnostic work.
- They must not be described or selected as active Stage2 mainline authority.
- Promoting such paths as Stage2 authority is an architecture contract
  violation.

### Layer 3: Audit boundary responsibilities
Audit occurs at the boundary between extracted candidates and publishable
database outputs. The audit boundary must expose:

- evidence pointers and span traceability
- field-level and formulation-level QC outcomes
- explicit conflict and uncertainty artifacts for targeted human review

Current governed interpretation:

- Layer 3 is not only an evaluation helper.
- It is also part of the production-grade audit and governance layer around the
  formulation database.
- The benchmark-valid endpoint remains the Stage 5 final formulation table.
- Reviewer-facing Layer 3 audit outputs remain downstream support surfaces and
  must not mutate benchmark-valid outputs.

Formulation-centered audit direction:

- the preferred reviewer entry object is one formulation row
- human review is split into two linked layers:
  - formulation existence and identity audit
  - value credibility audit
- these layers are not parallel:
  - value credibility depends on structure and identity correctness
  - many apparent value errors are projections of structure or identity errors
- current repo capability is partially present but not yet unified into one
  governed formulation-centered audit system contract

### Stable downstream deterministic rule families
The following rule families are considered stable deterministic core:

- formulation relation materialization
- numeric evidence realignment and token QC gating
- derivation and normalized field computation
- schema assembly and export formatting
- benchmark-facing final-table comparison
