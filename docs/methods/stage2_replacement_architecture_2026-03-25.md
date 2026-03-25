# Stage2 Replacement Architecture (2026-03-25)

## Status

This note records the approved direction for a true Stage2 replacement effort.
It does not switch the active benchmark runtime by itself.

Active benchmark-valid mainline remains:

- Stage2 -> Stage3 -> Stage5

Active Stage2 runtime remains:

- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`

Replacement scaffolding added in this pass:

- `src/stage2_sampling_labels/build_stage2_replacement_contract_v1.py`
- `data/db/db_v2/schema_manifest_v2_replacement.json`
- `data/db/db_v2/stage2_replacement_output_contract.tsv`
- `data/db/db_v2/stage2_legacy_to_replacement_mapping.tsv`

## Observed Repo Facts

### Current Stage2 runtime behavior

Directly observed from the current extractor, frozen active artifacts, and raw
responses:

- active authoritative Stage2 artifact is still a wide-row TSV/JSONL contract
- active frozen TSV has `119` columns
- current extractor code exposes `124` columns
- active frozen artifact still uses legacy `plga_mw_kDa_*`
- current code uses canonical `polymer_mw_kDa_*`
- Stage2 currently emits formulation identity plus fixed-slot material and
  measurement fields
- Stage2 also emits coarse evidence-oriented metadata such as:
  - `supporting_evidence_refs`
  - row-level evidence span fields
  - field `scope`
  - field `membership_confidence`
  - field `evidence_region_type`

### Current downstream expectations

Directly observed from Stage3 and Stage5:

- Stage3 expects a wide-row candidate TSV with identity fields plus `_value`
  families
- Stage3 dynamically scans `_value` columns and uses field scope and evidence
  region hints for relation materialization
- Stage5 still consumes wide-row identity and key material fields
- Stage5 accepts a legacy alias from `plga_mw_kDa` to `polymer_mw_kDa`
- current downstream code is not ready to consume object-first Stage2 payloads
  directly

### Constraint implied by current repo state

Inference:

- a clean replacement should not expand the legacy fixed-slot contract further
- Stage3 and Stage5 still need a compatibility surface while the redesign is
  underway

## Why The Old Stage2 Needed Replacement

The old Stage2 contract is useful for high-recall formulation discovery, but
it mixes together several concerns that should be separated:

- semantic extraction
- component representation
- measurement capture
- evidence handoff
- normalization-friendly values
- fixed-slot benchmark compatibility

Observed risks:

- fixed-slot material fields compress multi-component formulations
- polymer MW is partly still represented with PLGA-specific naming
- arbitrary DOE factors outside the current core list do not fit cleanly
- raw expression forms are only partially separated from parsed values
- Stage2 asks the LLM to do too much coarse evidence ownership work

## New Responsibility Boundary

### Stage2 LLM should do

- semantic object discovery
- formulation identity discovery
- component discovery
- phase discovery
- process-step discovery
- variable and factor discovery
- measurement discovery
- raw expression capture
- relation cue extraction
- coarse evidence handoff only

### Stage2 LLM should not do

- final fixed-slot normalization
- audit-grade evidence ownership binding
- definitive unit conversion
- derived value generation
- long arbitration narration
- final modeling-table filling

### Deterministic layers after Stage2 should do

- object consolidation
- normalization and canonicalization
- derived value generation
- exact evidence refinement and binding
- compatibility projection for current Stage3 and Stage5
- database assembly

## Replacement Object Model

The replacement Stage2 contract is organized around semantic candidate objects:

1. `formulation_identity_candidate`
2. `component_candidate`
3. `phase_candidate`
4. `process_step_candidate`
5. `variable_or_factor_candidate`
6. `measurement_candidate`
7. `relation_cue`
8. `evidence_handoff`

This object model is intentionally distinct from the downstream relational
database shape. Stage2 emits semantic candidates; deterministic post-processing
maps those candidates into `db_v2` entities.

## Recommended Implementation Shape

Recommended shape:

- a paired Stage2 semantic-emission script plus deterministic compatibility
  adapter

Reasoning:

- minimizes disruption to current Stage3 and Stage5 consumers
- avoids further growth of the legacy fixed-slot contract
- creates a clean landing zone for later `db_v2` migration
- allows semantic-object Stage2 work to proceed without pretending the current
  runtime is already object-native

Transitional rule:

- compatibility projection is required during migration
- compatibility projection is deterministic
- compatibility projection is transitional, not the long-term semantic source
  of truth

## Initial Implementation In This Pass

This pass adds design scaffolding only.

Implemented:

- a source-controlled replacement contract builder in `src/`
- a machine-readable replacement manifest
- a field-level replacement output contract TSV
- a legacy-to-replacement mapping TSV

Not implemented yet:

- object-native runtime extraction
- deterministic compatibility adapter over real replacement outputs
- Stage3 or Stage5 migration to object-native inputs
- active benchmark pipeline contract switch

## Migration Implications

Suggested order:

1. freeze the current wide-row Stage2 contract as the transition compatibility
   target
2. build object-native Stage2 emission beside the current runtime
3. build deterministic compatibility projection back to the current wide-row
   surface
4. validate Stage3 and Stage5 behavior against the compatibility projection
5. migrate downstream consumers only after compatibility is stable
6. retire fixed-slot assumptions after downstream adoption

## Non-Change Statement

- Stage2.5 remains archived and non-authoritative
- active benchmark runtime is not switched by this note
- no historical results were modified by this note
