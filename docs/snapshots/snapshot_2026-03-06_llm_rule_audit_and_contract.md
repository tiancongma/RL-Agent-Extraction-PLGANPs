# Snapshot: LLM Rule Audit and Responsibility Contract (2026-03-06)

## Project Objective
Extract auditable PLGA formulation data with reproducible downstream derivation, evaluation, and database export.

## Current Architecture in 3 Layers
1. LLM extraction layer: semantic structuring of formulation candidates.
2. Deterministic arbitration layer: normalization, derivation, evidence binding, schema/export.
3. Audit boundary layer: conflict queues, QC reports, and traceable evidence artifacts.

## Current Rule-vs-LLM Decision
- LLM owns semantic structure (instance boundaries, field-role assignment, shared-vs-instance interpretation).
- Deterministic layers own fact arbitration, numeric evidence checks, derivation, QC gating, and export.
- Semantic-repair-heavy downstream rules are now explicitly treated as upstream redesign candidates.

## What Changed This Week
- Added architecture clarification section to `project/2_ARCHITECTURE.md`.
- Appended formal decision entry dated 2026-03-06 in `project/4_DECISIONS_LOG.md`.
- Added method contract note:
  `docs/methods/llm_deterministic_responsibility_contract_2026-03-06.md`.

## Next Actions
1. Keep current deterministic release gates unchanged for run stability.
2. Prioritize extraction schema upgrades for the top semantic-compensation hotspots.
3. Measure redesign impact against existing audit outputs before retiring downstream semantic repair rules.
