# Stage2 Replacement Compatibility Adapter (2026-03-25)

## Purpose

This note records the first deterministic compatibility adapter for the true
Stage2 replacement effort.

Observed repo fact:
- Active benchmark runtime still flows through the legacy wide-row Stage2
  surface into Stage3 and Stage5.
- The approved replacement direction is semantic-object Stage2 output, not
  continued expansion of the fixed-slot Stage2 contract.

Decision in this pass:
- Add a deterministic transitional adapter that reads semantic-object Stage2
  payloads and projects them back into the legacy wide-row Stage2 surface.

Non-decision:
- This adapter does not make the semantic-object contract the active benchmark
  runtime by itself.

## Why The Adapter Is Needed

- Stage3 and Stage5 currently depend on the legacy Stage2 wide-row surface.
- We want the replacement Stage2 to become operational without forcing another
  round of fixed-slot-first LLM design.
- A deterministic bridge keeps the new semantic core clean while preserving
  current downstream compatibility during migration.

## Inputs

Primary input:
- semantic-object Stage2 JSONL payloads with these object families:
  - `formulation_identity_candidate`
  - `component_candidate`
  - `phase_candidate`
  - `process_step_candidate`
  - `variable_or_factor_candidate`
  - `measurement_candidate`
  - `relation_cue`
  - `evidence_handoff`

Implementation:
- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`

## Outputs

Compatibility artifacts:
- `weak_labels__v7pilot_r3_fixparse.tsv`
- `weak_labels__v7pilot_r3_fixparse.jsonl`
- `compatibility_projection_trace_v1.tsv`
- `compatibility_projection_summary_v1.json`

Contract artifact:
- `data/db/db_v2/stage2_replacement_compatibility_projection_contract.tsv`

## Projection Rules

Direct projection:
- formulation identity fields from `formulation_identity_candidate`
- component names into legacy component slots
- process names into `emul_method` and `preparation_method`
- measurement values by deterministic measurement-name matching
- coarse evidence locator fields from `evidence_handoff`

Derived projection:
- `polymer_identity` from polymer component names
- `emul_type` from phase candidates and coarse factor hints
- `*_scope` from repeated row values within one document
- `*_membership_confidence` from projection status, not from new LLM output

Unavailable or partial projection:
- fields remain blank when the replacement objects do not contain enough
  deterministic information
- `*_missing_reason` records that the value was not projectable from the
  current replacement payload
- exact evidence ownership binding is intentionally not invented here

## Transitional Limits

- Multiple same-role components are compressed into pipe-delimited legacy text
  values when necessary for downstream compatibility.
- The adapter does not normalize units into canonical scientific forms.
- The adapter does not perform final evidence arbitration.
- Legacy field names such as `plga_mass_mg` remain only as transitional output
  names for Stage3 and Stage5 compatibility.

## Governance Status

- Transitional deterministic support infrastructure
- Not an active benchmark entrypoint by itself
- Compatible with the approved Stage2 replacement architecture direction
- Historical results under `data/results/` remain untouched by this method note
