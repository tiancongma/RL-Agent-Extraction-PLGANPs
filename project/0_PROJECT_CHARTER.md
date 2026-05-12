# Project Charter

RL-Agent-Extraction-PLGANPs exists to build an auditable PLGA formulation
extraction pipeline from scientific literature into structured
formulation-level records.

## Purpose

The project converts governed literature inputs into traceable formulation
tables. It prioritizes reproducibility, explicit provenance, and reviewable
evidence over hidden orchestration or one-off extraction shortcuts.

## Scope

The project covers:

- PLGA nanoparticle formulation literature.
- Zotero-derived corpus intake, cleaned content, table assets, and scoped
  manifests.
- LLM semantic formulation identification from governed evidence surfaces.
- Deterministic relation materialization, validation, audit, final-table
  closure, and downstream review surfaces.
- GT comparison as a diagnosis-baseline tool for measuring current extraction
  behavior against fixed human reference assets.

The project does not cover wet-lab validation, biological outcome modeling,
automatic hypothesis generation, or uncontrolled external data integration.

## Authority

Authoritative project contracts live in `project/`. Execution-facing pipeline
logic lives in `src/`. Datasets and run artifacts live in `data/`. Supporting
documentation, audits, plans, and historical explanations live in `docs/`.

Active sources must be resolved explicitly. Agents and scripts must not infer
current inputs from recency, filename similarity, directory sorting, or memory.

## Design Intent

The LLM owns semantic formulation discovery. Deterministic code owns
validation, relation handling, normalization, evidence checking, audit, and
materialization within the semantic scope already authorized by the pipeline.

Stage outputs are useful only when their boundary contracts are explicit.
Diagnosis baselines and GT comparisons are expected development tools; they do
not change the production-stage responsibilities.

## Success Criteria

The project succeeds when formulation outputs are reproducible, tabular,
traceable to governed source evidence, reviewable by humans, and stable enough
to support iterative diagnosis without changing the underlying authority
contracts.
