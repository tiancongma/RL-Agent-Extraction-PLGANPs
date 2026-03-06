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

## Stage 5 — Results and Publication

### Purpose
Aggregate extraction outputs into publishable datasets.

### Key Principle
Results are organized by **run_id**, not by vague version names.

### Location
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
project/3_STATE_MACHINE.md

- Architectural decisions are recorded in:
project/4_DECISIONS_LOG.md

- Agent run procedure: see
project/6_AGENT_RUNBOOK.md

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
