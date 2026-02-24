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

## Current Data State

Based on latest coverage summary:
- 158 formulation-level EE records
- 89 records with loading proxy
- 68 records with polymer identity + loading proxy
These numbers define the modeling-ready core subset.
This branch treats these records as:
Evidence-grounded
Structurally de-duplicated
Schema_v3 compliant

## Non-Goals

This branch does not:
- Redesign extraction prompts
- Modify schema definitions
- Change stage-level architecture
- Introduce new document parsing mechanisms
PDF structural recovery is out of scope.

## Transition Criteria

This branch will be considered complete when:
- EE coverage statistics are stable
- Modeling-ready dataset is frozen
- Documentation clearly separates extraction layer from modeling layer

## Guidance for Automated Agents

Before making structural changes:
- Read PROJECT_ARCHITECTURE.md
- Read README.md
- Confirm that modifications do not alter stage responsibilities

All modeling scripts should be implemented outside core extraction stages.