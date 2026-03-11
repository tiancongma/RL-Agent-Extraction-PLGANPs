Branch: feature/ee-coverage-rl
Status: Active
Scope: Encapsulation Efficiency coverage analysis and modeling preparation

## Purpose of This Branch

This branch focuses on Encapsulation Efficiency (EE) coverage consolidation and preparation for interpretable modeling.
The goal is not to modify the core extraction architecture, but to:
- Quantify formulation-level EE coverage
- Stabilize evidence-grounded EE records
- Identify loading proxies and polymer identity fields
- Prepare a high-confidence dataset for downstream modeling

All core pipeline design remains governed by PROJECT_ARCHITECTURE.md.

---

## 🔁 Scope Update (Extraction Upgrade Validation Mode)

As of the current development phase, this branch additionally serves as a **controlled validation environment for upgrading general extraction logic**.

This does NOT change the architectural contracts defined in `project/2_ARCHITECTURE.md`.

However, the branch is now permitted to:

- Improve Stage1 cleaning robustness
- Improve Stage2 weak-label extraction behavior
- Additive schema extensions (new optional columns only)
- Strengthen evidence grounding logic
- Harden canonicalization and normalization rules
- Introduce deterministic run_id behavior enforcement

These upgrades must:

- Preserve canonical directory structure
- Preserve manifest and key2txt contracts
- Remain backward-compatible with Stage definitions
- Not introduce breaking schema changes

Any extraction upgrade implemented here must remain general-purpose and not EE-specific.

---

## Relationship to Main Pipeline

This branch reuses the stable extraction pipeline:

Stage0 – Relevance  
Stage1 – Cleaning  
Stage2 – Extraction  
Stage3 – Derivation  
Stage4 – Evaluation  
Stage5 – Benchmark  

No architectural restructuring is performed here.

All changes in this branch operate on:

- Extracted results
- Evidence-aligned fields
- Coverage statistics
- Dataset consolidation logic

Extraction upgrades performed here must remain compatible with mainline pipeline stages and are candidates for future merge-back to `main` after validation.

---

## Current Data State

Based on latest coverage summary:
- 158 formulation-level EE records
- 89 records with loading proxy
- 68 records with polymer identity + loading proxy

These numbers define the modeling-ready core subset.

This branch treats these records as:

- Evidence-grounded
- Structurally de-duplicated
- Schema_v3 compliant

---

## Non-Goals

This branch does not:

- Perform directory restructuring
- Break canonical data contracts
- Introduce stage-level redesign
- Modify manifest or key2txt semantics
- Introduce EE-only logic into core extraction scripts

PDF structural recovery remains out of scope.

---

## Transition Criteria

This branch will be considered complete when:

- EE coverage statistics are stable
- Modeling-ready dataset is frozen
- Extraction logic upgrades are validated
- General-purpose extraction improvements are ready for selective merge-back
- Documentation clearly separates extraction layer from modeling layer

---

## Merge Candidate Policy

The following types of changes may be merged back to `main`:

- Stage1 cleaning improvements
- Stage2 extraction robustness upgrades
- Additive schema columns (non-breaking)
- Deterministic run_id enforcement utilities
- Evidence hardening logic

The following must remain branch-only:

- Goren-specific alignment scripts
- EE-only coverage filters
- Benchmark-specific diagnostics
- Modeling datasets
- Data under `data/benchmark/` and `data/db/`

All merge-back candidates must be explicitly reviewed and recorded before integration into `main`.

---

## Guidance for Automated Agents

Before making structural changes:

- Read `project/2_ARCHITECTURE.md`
- Read `README.md`
- Confirm modifications do not alter stage responsibilities
- Confirm whether the change is:
  - General extraction upgrade (future merge candidate)
  - EE-specific benchmark logic (branch-only)

All modeling scripts must remain outside core extraction stages.