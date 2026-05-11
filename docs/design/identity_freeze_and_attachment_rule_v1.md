# Identity Freeze And Attachment Rule v1

Status:
- active engineering contract
- enforced through a hard gate at the Stage5 post-materialization boundary

## Motivation

DEV15 reviewed-boundary work established that formulation identity can be
treated as stable for the intended benchmark scope.

The remaining risk is downstream:
- Stage3 / Stage5 / compare surfaces can still bind on unstable final-row ids
- new identity-related variables can enrich rows but may appear to degrade
  evaluation if downstream tooling treats attribute changes as identity changes
- value-level or measurement-level signals can accidentally reshape identity
  when the correct behavior is to attach values to an existing formulation

This contract hardens the pipeline around one principle:

> Once formulation identity is frozen, downstream stages must attach values,
> not reconstruct formulations.

## Identity Freeze

Identity freeze means:
- formulation count is fixed for the declared frozen scope
- formulation membership is fixed for the declared frozen scope
- downstream stages may enrich, resolve, and derive fields on those identities
- downstream stages may not silently recompose the identity set

For reviewed scopes, the identity authority is the frozen Layer2-style
formulation boundary surface.

## Attribute Attachment

Attribute attachment means downstream work may:
- add fields
- resolve missing fields
- derive fields
- preserve extra identity-bearing metadata such as `identity_variables_json`

Attribute attachment does not allow:
- implicit split
- implicit merge
- new formulation creation from value similarity

## IDENTITY_FREEZE_RULE_V1

After Layer2 identity, or an equivalent frozen boundary authority, the
following must hold:

1. formulation count remains invariant
2. formulation membership remains invariant

Downstream stages may:
- add fields
- resolve missing fields
- derive fields

Downstream stages must not:
- split formulations implicitly
- merge formulations implicitly
- create new formulations based on value similarity

## Split Authorization Policy

Field classes:

### identity_defining_fields

Examples:
- reviewed article-native formulation labels
- reviewed family/core/variant boundaries
- explicitly authorized identity-bearing variables

Only these fields may trigger an identity split, and only when the split is:
- paper-local
- traceable
- explicitly authorized by the frozen identity policy

### non_identity_fields

Examples:
- helper annotations
- display labels
- reviewer convenience fields

Default rule:
- attach to the existing identity

### measurement_fields

Examples:
- size
- PDI
- zeta
- EE
- LC

Hard rule:
- measurement fields must never trigger identity split by default

Uncertainty rule:
- if uncertain, attach to the existing identity

## Interaction With Existing Active Surfaces

### `identity_variables_json`

- `identity_variables_json` may preserve identity-bearing variables across the
  active semantic Stage2 -> adapter -> Stage3 -> Stage5 chain
- it is not a license for downstream stages to recompose the frozen identity
  set automatically
- once identity is frozen, the carrier should enrich the row and support audit,
  not silently expand formulation membership

### collapse signature

- collapse signatures may help distinguish rows during materialization
- they are not by themselves an authority to override a frozen reviewed
  identity scaffold

### final table materialization

- Stage5 remains benchmark-valid materialization
- final display ids and namespaced representative ids are not the frozen
  identity authority for downstream compare

## Forbidden Behaviors

- binding GT or audit surfaces directly on unstable final display ids
- letting measurement fields reshape formulation identity
- allowing value-level similarity to create or remove formulation rows
- using coarse fallback matching as benchmark-grade identity binding

## Guardrail Insertion Point

This rule is enforced at the Stage5 post-materialization boundary through a
hard gate:

- input:
  - upstream identity scaffold surface
  - Stage5 final table
- checks:
  - row count drift
  - identity reassignment
  - unresolved or ambiguous bindings
- action:
  - emit diagnostics
  - fail the run with non-zero exit status on any violation
  - do not silently fix

If identity freeze is violated, the run is invalid and must not proceed to:
- value-level evaluation
- audit-ready export
- Layer3 field GT evaluation

This keeps the active pipeline stable while making identity violations both
visible and blocking.

## Non-Goals

- not changing Stage2 extraction semantics
- not redesigning identity discovery
- not modifying core Stage3 relation logic
- not changing benchmark-valid Stage5 outputs in place
- not introducing fuzzy matching or value-based regrouping
