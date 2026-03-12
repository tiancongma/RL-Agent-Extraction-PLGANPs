# Architecture

This document defines the **conceptual and data architecture** of the project.
It specifies the stable structure of the pipeline, the responsibilities of each
stage, and the data contracts between stages.

This file is intentionally conservative and should change only when the
**project scope changes**, not when individual scripts, prompts, or models change.

---

## Design Philosophy

This project is organized around **data semantics**, not script order.

- Scripts may change
- Prompts may change
- Models may change

However, the **data boundaries, directory semantics, and single sources of truth
must remain stable**.

The goal is to ensure that:
- Results are reproducible
- Experimental variation is explicit
- “Latest” is never inferred from file names or memory

---

## Pipeline Overview

The pipeline is divided into conceptual stages.
Each stage produces artifacts with well-defined semantics and lifecycle rules.

Stages are **conceptual**, not tied 1:1 to scripts.
The active implementation namespaces are restricted to `Stage 0` through
`Stage 5` only.

---

## Stage 0 — Raw Metadata and Relevance Filtering

### Purpose
Identify candidate papers that are potentially relevant to PLGA nanoparticle formulation.

### Typical Operations
- Regex-based pre-filtering
- LLM-based relevance classification
- Zotero tagging and snapshot fetching

### Characteristics
- Outputs are re-runnable
- Outputs are non-binding
- Downstream stages must not depend on these files directly

### Location
data/raw/zotero/

---

## Stage 1 — Cleaned Content and Manifest

### Purpose
Convert HTML/PDF documents into cleaned, structured text and establish a stable
mapping between papers and extracted content.

### Key Outputs
- Cleaned full text
- Section-level representations
- Table-level representations
- Manifest linking paper keys to content

### Single Source of Truth
data/cleaned/index/manifest_current.tsv

### Location
data/cleaned/content/
data/cleaned/index/

### Invariants
- Any change to files in `data/cleaned/index/` requires re-running all downstream stages
- Only one `manifest_current.tsv` may exist

---

## Stage 2 — Sampling and Weak Label Extraction

### Purpose
Define experimental subsets and generate weak labels using LLMs.

### Key Artifacts
- Sample definitions (e.g. sample10, sample20)
- Key-to-text index
- Versioned weak labels

### Single Source of Truth
data/cleaned/index/key2txt.tsv

### Location
data/cleaned/samples/
data/cleaned/labels/weak/

---

## Stage 3 — Manual Annotation (Ground Truth)

### Purpose
Provide partial, human-curated labels for evaluation.

### Characteristics
- Ground truth is intentionally incomplete
- Manual labels do not overwrite weak labels
- Disagreement between weak and manual labels is expected

### Location
data/cleaned/labels/manual/

---

## Stage 4 — Evaluation and Optimization

### Purpose
Quantitatively evaluate extraction quality and support iterative improvement of
prompts or parsers.

### Typical Operations
- Rule-based grading
- Metric calculation (accuracy, hallucination rate, coverage)
- Optional RL-assisted optimization

---

## Stage 5 — Final Formulation Closure And Benchmark Comparison

### Purpose
Convert candidate formulation-instance outputs into final one-row-per-
formulation records and compare only those final records to GT.

### Key Principle
Stage 5 is the only benchmark-valid reporting layer. Earlier stages may produce
diagnostic comparisons, but they are not the official system result.

### Location
src/stage5_benchmark/
data/results/run_<run_id>/

---

## Run Tracking and Reproducibility

Each pipeline execution is assigned a unique run_id:

run_YYYYMMDD_HHMM_<git_commit>_<sample>

The most recent valid run is recorded in:
runs/latest.txt

This is the **only authoritative definition of “latest”**.

---

## Architectural Invariants (Do Not Break)

- Files in `data/cleaned/index/` are pipeline-critical
- `manifest_current.tsv` and `key2txt.tsv` are unique
- No script may silently depend on legacy artifacts
- Experimental variation must be expressed via:
  - run_id
  - configuration
  - git commit

---

## Relationship to Project Governance

- Project scope and stage transitions are defined in:
this document

- Architectural decisions are recorded in:
project/4_DECISIONS_LOG.md

- Agent run procedure: see
project/ACTIVE_PIPELINE_RUNBOOK.md

- Current active script registry and DEV execution path: see
project/ACTIVE_PIPELINE_RUNBOOK.md

This architecture is intentionally minimal, explicit, and extensible.

---

## Debug & Human Review Contract

All debug artifacts intended for manual inspection MUST include DOI-level metadata.

Invariant:
Any Excel or TSV generated for human review must contain:

- doc_key (Zotero key)
- reference_normalized_doi (DOI)
- doi_url (https://doi.org/<DOI>)
- paper_title (if available)
- publication_year (if available)

Rationale:
Manual verification requires immediate access to original publications.
DOI is the canonical external identifier and must be included.

Scope:
Applies to:
- benchmark debug matrices
- patch_queue outputs
- per-doc regression diagnostics
- modeling-ready merged instance summaries
- any file labeled debug, audit, review, or manual

Non-negotiable:
No debug artifact is considered complete if DOI is missing.

---

## LLM Extraction Layer, Deterministic Arbitration Layer, and Audit Boundary

### Why the pipeline is layered
The project separates semantic extraction, deterministic arbitration, and audit so that:
- semantic interpretation is handled where language context is strongest (LLM stage),
- reproducible rule behavior is handled where strict consistency is required (deterministic stage),
- human and machine audit can verify both without mixing responsibilities.

This prevents hidden logic drift where semantic decisions are silently embedded into late-stage scripts.

### Layer 1: LLM extraction responsibilities
The LLM extraction layer is responsible for:
- identifying formulation instances and instance boundaries,
- assigning field role semantics (what value belongs to which field),
- distinguishing shared-vs-instance-specific meaning when reported in prose/tables,
- emitting structured candidate rows with explicit missingness rather than silent omission.

The LLM extraction layer is not responsible for final arbitration of conflicting evidence.

### Layer 2: Deterministic arbitration responsibilities
Deterministic scripts are responsible for:
- numeric evidence binding and token-level support checks,
- deterministic derivation and unit/ratio normalization,
- schema assembly and export formatting,
- stable filtering/gating for reproducible benchmarking and release outputs.

These responsibilities must remain deterministic to preserve run-to-run reproducibility and auditability.

### Layer 3: Audit boundary responsibilities
Audit occurs at the boundary between extracted candidates and publishable database outputs.
The audit boundary must expose:
- evidence pointers and span traceability,
- field-level and formulation-level QC outcomes,
- explicit conflict/uncertainty artifacts for targeted human review.

### Stable downstream rule families (intentionally deterministic core)
The following rule families are considered stable deterministic core:
- numeric evidence realignment and token QC gating,
- derivation and normalized field computation,
- schema table construction and database export,
- deterministic multi-model merge/conflict reporting.

### Downstream rule families considered upstream-redesign candidates
The following rule families are currently tolerated but should be treated as future upstream LLM redesign targets:
- semantic formulation grouping and regrouping,
- global baseline inheritance for shared method conditions,
- drug/surfactant semantic normalization used to recover membership,
- condition-instance key inference and shared-vs-instance reconstruction,
- repeated semantic repair logic that compensates for weak extraction schema structure.

---

## Formulation Extraction Pipeline Structure (Current)

### System Goal
Build an auditable PLGA nanoparticle formulation database for EE modeling.

### Core Output Constraint
The final project output must remain a tabular dataset with one row per formulation.

### Why an intermediate representation is required
Paper structure does not map directly to a final formulation row. Methods, tables, and results may mix shared conditions, inherited settings, and instance-specific values. Therefore the pipeline must assemble formulation records through intermediate representations before final flattening.

### Current system interpretation
The project should now be read as a formulation-instance-centered, literature-evidence-grounded, multi-layer system rather than as a pure field-extraction pipeline. The front of the system is optimized for candidate-instance recall and explicit evidence capture. The middle of the system evaluates candidate-instance behavior and exposes where instance boundaries are preserved, collapsed, or over-expanded. The back of the system is responsible for light guardrail filtering, normalization where explicitly adopted, and eventual production of one-row-per-formulation outputs for downstream modeling.

This interpretation is important because the same paper can legitimately have more candidate-instance rows than final formulation-table rows. Baseline labels, optimized labels, parent/variant language, and sweep-style reporting are provenance about how the paper expresses identity. They are not by themselves a permanent contract that forbids later comparison or collapse by core formulation parameters when a downstream layer explicitly owns that work.

### Eight-layer formulation-instance architecture

#### 1) Corpus and document assets layer
This layer owns stable corpus membership and document identity. It includes Zotero keys, DOI-linked manifests, dataset-scoped content roots, and fixed split definitions such as DEV subsets. Its job is to keep paper identity, subset membership, and reproducible inputs stable before any extraction occurs.

#### 2) Document cleaning and structural normalization layer
This layer converts PDF and HTML sources into cleaned text, sections, and table assets. The purpose is not final formulation reasoning; it is to preserve enough document structure that later evidence packing and traceability remain possible. Structural cleanup belongs here because later stages depend on stable section and table anchors rather than raw source files.

#### 3) Evidence block construction and packing layer
Before the LLM sees a paper, the system assembles a bounded evidence view from metadata, paragraphs, captions, and table-derived material. This layer controls ordering, inclusion, and emphasis, and therefore strongly influences whether candidate formulation instances are enumerated correctly. It is distinct from cleaning because it is model-facing assembly rather than source normalization.

#### 4) LLM formulation-instance extraction layer
This layer proposes candidate formulation instances with high recall. It emits instance-centric rows with routing semantics such as `new_formulation`, `variant_formulation`, `parent_instance_id`, `change_role`, and evidence references. It owns semantic interpretation of formulation boundaries and inherited-vs-overridden synthesis meaning, but it does not by itself guarantee final database closure.

#### 5) Candidate formulation graph layer
The natural intermediate representation after extraction is not just a flat table of fields; it is a candidate formulation graph. Candidate nodes may stand in parent/variant, baseline/optimized, shared-method, or sweep-style relationships. This is the layer where under-segmentation and over-segmentation become meaningful system behaviors rather than simple field errors.

#### 6) Instance-level evaluation and diagnostic layer
This layer compares candidate-instance behavior against benchmark expectations and review artifacts. Count difference, under-segmentation, over-segmentation, per-paper mismatch diagnostics, and reviewer workbooks all belong here. The layer evaluates candidate-instance structure, not only scalar field correctness.

#### 7) Light guardrail and final formulation filtering layer
This layer owns precision recovery before final export. It may remove obvious non-formulation rows, suppress clearly redundant rows, and collapse rows that differ only in non-core measurement dimensions when an explicit contract says to do so. It should remain lighter than the older rule-heavy reconstruction systems preserved in `archive/code/`, and it must not be confused with Stage2 candidate generation itself.

#### 8) Final formulation table and modeling layer
This is the database-facing layer: one row per final formulation with modeling-ready columns for EE, LC, size, and formulation parameters. The final modeling table is downstream of candidate extraction and downstream of any explicitly adopted guardrail or normalization logic. This layer is the place where release-oriented filtering decisions such as PLGA-only modeling subsets belong.

### Current stage flow for formulation assembly

#### 1) Document preprocessing
Input documents (PDF/HTML) are normalized into cleaned text blocks, section artifacts, and extracted table artifacts. These artifacts establish deterministic anchors for downstream evidence tracing.

#### 2) LLM semantic extraction
The LLM layer performs semantic understanding: formulation instance detection, candidate field extraction, and inheritance interpretation (including shared-vs-instance-specific meaning).

#### 3) Formulation hypothesis layer
The system materializes candidate formulation records (hypotheses) prior to hard verification. These records may be incomplete, conflicting, or uncertain by design.

#### 3a) Formulation-instance routing contract
Formulation hypotheses must stay formulation-centric and instance-aware. The extraction layer should emit candidate formulation instances, not detached field fragments that are grouped only later.

Primary instance routing enums are fixed to:
- `new_formulation`
- `variant_formulation`
- `candidate_non_formulation`
- `unclear`

Primary change-role enums are fixed to:
- `synthesis_defining`
- `non_synthesis`
- `unclear`

Interpretation rules:
- `new_formulation`: a distinct formulation row candidate that does not require parent-based inheritance to establish identity.
- `variant_formulation`: a distinct formulation row candidate defined relative to a parent/base formulation through inheritance or explicit comparative change language.
- `candidate_non_formulation`: a mentioned variant/condition/context that should not become a formulation row by default unless later evidence upgrades it.
- `unclear`: evidence is insufficient for confident routing.

Critical boundary rule:
- Post-processing differences, test conditions, storage conditions, release-test conditions, and characterization/measurement contexts do not automatically define new formulation rows when synthesis-defining parameters are unchanged.

Identity rule:
- Formulation identity is not limited to a frozen core schema. Any true synthesis/design variable reported in the paper can be identity-defining even if it sits outside the current core field list.

Auxiliary tags such as `doe`, `sweep`, `post_processing`, `test_condition`, `measurement_context`, `optimized`, and `control` may be stored in `instance_context_tags` / `change_context_tags`, but they are not primary routing enums.

#### 4) Evidence binding
Deterministic logic binds hypothesis fields to local evidence (document spans and/or table cells), with explicit evidence pointers and reproducible offsets.

#### 5) Formulation-level audit
Audit verifies that fields grouped under one formulation are mutually consistent and belong to the same instance boundary, exposing conflict and uncertainty artifacts for targeted review.

#### 6) Final tabular export
After verification and audit gates, formulation records are flattened into modeling-ready tabular outputs while preserving traceability fields needed for reproducibility.

### Role separation (non-negotiable)

#### LLM responsibilities
- semantic understanding of scientific prose/tables
- formulation instance detection and boundary interpretation
- inheritance interpretation for shared vs instance-specific conditions

#### Rule-based system responsibilities
- evidence localization
- numeric matching and support checks
- normalization and deterministic derivation
- verification, gating, and export assembly

### Representation boundary
Internal representations may contain richer structures (candidate hypotheses, evidence bundles, conflict queues, trace artifacts), but the released database remains tabular.

### Benchmark comparison and normalization contract
- Stage2 formulation-instance outputs are candidate formulation rows, not generic formulation-core rows.
- Generic collapse by core formulation parameters belongs to downstream normalization/schema layers when such a layer is explicitly part of the selected workflow.
- The current active DEV-15 formulation-instance comparison path does not insert a generic normalization/core-signature layer between Stage2 extraction and Stage4 counting.
- The fixed DEV15 formulation-skeleton workbook is a benchmark input artifact for that active path.
- The script family that originally bootstrapped the skeleton workbook lives under `archive/code/dev15_skeleton_bootstrap/` and is historical benchmark-preparation tooling, not an active runtime normalization layer.
- Any paper-specific reconciliation used in the active DEV-15 path must be explicit, documented, and localized in the active evaluator contract rather than assumed from historical Stage5 or archived rule-heavy paths.

### Current ownership boundary
Under the current documented system, candidate extraction, candidate-graph behavior, instance-level evaluation, light guardrail filtering, and final formulation-table semantics are separate concerns even when the active path only implements part of that stack. The current DEV-15 path implements candidate extraction and candidate-instance evaluation directly against the fixed skeleton benchmark, with only narrow documented reconciliation. It does not yet implement a generic light-normalization layer for final formulation closure. If the repository later chooses to compare predictions against final formulation-table targets instead of candidate-instance targets, that guardrail/normalization contract must be explicitly defined and wired into the active path rather than inferred from archived skeleton bootstrap scripts or Stage5 benchmark tooling.

### Minimal final-output layer and benchmark-valid closure path
The repository has a durable design contract for the minimal final-output layer
at `project/design/MINIMAL_FINAL_OUTPUT_LAYER_DESIGN.md`. The active phase-1
implementation now lives inside the single Stage 5 namespace under
`src/stage5_benchmark/`. In the current active structure, Stage 5 owns both:

- conservative final formulation-table closure from Stage 2 candidate rows
- final-table-vs-GT comparison

This does not create a hidden end-to-end orchestrator. The canonical full path
remains the manual Stage 0 to Stage 5 sequence defined in
`project/ACTIVE_PIPELINE_FLOW.md`.

## Repository Benchmark Policy

Formal GT comparison belongs only to the full pipeline final-output layer. This repository is multi-layer by design, so intermediate layers and final formulation-table outputs do not share identical semantics. Candidate-instance rows, candidate graphs, packing audits, and component-level regression artifacts may be inspected against GT for diagnosis, ablation, and failure localization, but those outputs are not benchmark-valid final results.

This rule is non-optional. If a workflow intends to include downstream guardrail filtering, normalization, or final formulation collapse before release, those layers must be executed before official GT comparison is reported. Candidate-instance layer semantics and final formulation-table semantics must not be conflated. Baseline or optimized labels may remain useful provenance at intermediate layers, but they do not convert an intermediate artifact into a final benchmark output.

The practical consequence is that direct Stage2-vs-GT or partial-path-vs-GT
comparisons are diagnostic only unless the repository explicitly defines that
partial path as the complete benchmark contract. In the current repository, the
complete benchmark contract is the manual Stage 0 to Stage 5 path culminating
in the Stage 5 final formulation table and its GT comparison outputs.
Intermediate counts may still be examined, but they must be labeled as
non-final and non-benchmark-valid.

---

## Consolidated Project Lifecycle States

This section consolidates the former state-machine governance content.

### STATE_0: Scope Locked

- Goal, scope, and non-goals are fixed.
- Allowed: documentation updates and minor refactoring without behavior change.
- Forbidden: adding new objectives or expanding problem scope.

### STATE_1: Data Ready

- Raw and cleaned data pipelines are functional and stable.
- Allowed: regenerating raw or cleaned data and fixing parsing bugs.
- Forbidden: changing downstream evaluation logic.

### STATE_2: Pipeline Frozen

- The extraction pipeline structure is frozen; work is limited to stability and correctness.
- Allowed: bug fixes, robustness improvements, and prompt refinements without schema expansion.
- Forbidden: adding new extraction fields, changing schemas, or introducing new models by default.

### STATE_3: Results Stable

- Outputs and evaluation metrics are stable enough for reporting.
- Allowed: figure generation, table generation, and result summarization.
- Forbidden: pipeline logic changes or data regeneration.

### STATE_4: Writing and Release

- Focus shifts to writing, sharing, and publication.
- Allowed: release-oriented documentation and manuscript preparation.
- Forbidden: new experiments and pipeline changes.

---

## Consolidated Historical Specification Contracts

This section preserves durable architectural content from the historical
`project_specification_UPDATED_*` files.

### Stable Repository Interface

- Repository directory structure is treated as a stable interface.
- Additive growth is allowed; path-breaking relocation is not the default change mode.

### Modular Governance Rule

- Architecture, requirements, run discipline, and active execution path are governed by separate authoritative files.
- No single monolithic specification file should override the modular governance stack.
